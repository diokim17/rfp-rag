"""나상훈: 원본 파일 → Document 목록. CSV 열 이름은 metadata에 유지합니다."""

import csv
import hashlib
import json
import re
import struct
import unicodedata
import zlib
from pathlib import Path

import olefile
import pymupdf

from observability import observed


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# HWP5 제어문자: 1유닛짜리(0,10,13,24~31) 외의 0~31은 8유닛(16바이트)을 차지하므로 통째로 건너뜀
CHAR_CTRL = {0, 10, 13} | set(range(24, 32))
HWPTAG_PARA_TEXT = 67
HWPTAG_CTRL_HEADER = 71
HWPTAG_LIST_HEADER = 72
HWPTAG_TABLE = 77


def _para_text(b: bytes) -> str:
    units = struct.unpack_from(f"<{len(b) // 2}H", b)
    out, i = [], 0
    while i < len(units):
        c = units[i]
        if c >= 32:
            out.append(c)
            i += 1
        elif c in CHAR_CTRL:
            if c == 10:
                out.append(10)
            i += 1
        else:
            if c == 9:  # 탭
                out.append(32)
            i += 8
    # 서로게이트 쌍(이모지 등)을 살리려고 코드 유닛을 모아서 한 번에 디코딩
    return struct.pack(f"<{len(out)}H", *out).decode("utf-16-le", "ignore")


def _records(data: bytes):
    """HWP5 레코드를 (tag, level, payload)로 순회. level이 깊을수록 안쪽(표 셀 등)."""
    i = 0
    while i < len(data):
        header = struct.unpack_from("<I", data, i)[0]
        tag, level, size = header & 0x3FF, (header >> 10) & 0x3FF, (header >> 20) & 0xFFF
        i += 4
        if size == 0xFFF:
            size = struct.unpack_from("<I", data, i)[0]
            i += 4
        yield tag, level, data[i:i + size]
        i += size


def _render_table(rows: int, cols: int, cells: list, nested: bool = False) -> str:
    """cells [(row, col, rowspan, colspan, text)] → 마크다운 표.
    세로 병합 셀은 값을 반복해서 행 하나만 떼어 봐도 뜻이 통하게 함 (청킹 대비)."""
    grid = [[""] * cols for _ in range(rows)]
    for r, c, rs, cs, text in cells:
        for rr in range(r, min(r + rs, rows)):
            grid[rr][c] = text
    lines = [row for row in grid if any(row)]
    if not lines:
        return ""
    keep = [j for j in range(cols) if any(row[j] for row in lines)]  # 빈 열(여백용) 제거
    lines = [[row[j] for j in keep] for row in lines]
    if nested:  # 셀 안의 표는 바깥 표를 깨지 않게 한 줄로
        return " / ".join(" · ".join(x for x in row if x) for row in lines)
    if len(keep) == 1 or len(lines) == 1:  # 1열 표(글상자)·1행 표(제목 장식) → 일반 문단
        return "\n".join(" ".join(row) for row in lines)
    out = ["| " + " | ".join(row) + " |" for row in lines]
    out.insert(1, "|" + "---|" * len(keep))
    return "\n".join(out)


def _section_text(data: bytes) -> list[str]:
    out, stack = [], []  # stack: 열려 있는 표들 (셀 안에 표가 중첩될 수 있음)

    def emit(text):
        if stack and stack[-1]["cur"] is not None:
            stack[-1]["cur"].append(text)
        else:
            out.append(text)

    def close():
        t = stack.pop()
        cells = [(r, c, rs, cs, _cell_text(" ".join(texts)))
                 for r, c, rs, cs, texts in t["cells"]]
        text = _render_table(t["rows"], t["cols"], cells, nested=bool(stack))
        if text:
            emit(text)

    for tag, level, b in _records(data):
        while stack and level <= stack[-1]["level"]:  # 표 레벨 이하로 돌아오면 표 끝
            close()
        top = stack[-1] if stack else None
        if tag == HWPTAG_CTRL_HEADER and b[:4] == b" lbt":  # ctrl id 'tbl ' (리틀엔디언)
            stack.append({"level": level, "rows": 0, "cols": 0, "cells": [], "cur": None})
        elif top and level == top["level"] + 1 and tag == HWPTAG_TABLE:
            top["rows"], top["cols"] = struct.unpack_from("<HH", b, 4)
        elif top and level == top["level"] + 1 and tag == HWPTAG_LIST_HEADER:
            c, r, cs, rs = struct.unpack_from("<4H", b, 8) if len(b) >= 16 else (0, 0, 0, 0)
            if r < top["rows"] and c < top["cols"]:
                top["cells"].append((r, c, max(rs, 1), max(cs, 1), []))
                top["cur"] = top["cells"][-1][4]
            else:  # 캡션 등 셀이 아닌 리스트 → 표 밖 문단으로
                top["cur"] = None
        elif tag == HWPTAG_PARA_TEXT:
            emit(_para_text(b))
    while stack:
        close()
    return out


def parse_hwp(path: str) -> str:
    """HWP5 BodyText 섹션의 문단 텍스트를 순서대로 추출. 표는 마크다운 표로 복원."""
    with olefile.OleFileIO(path) as f:
        compressed = f.openstream("FileHeader").read()[36] & 1
        sections = sorted(
            (s for s in f.listdir() if s[0] == "BodyText"),
            key=lambda s: int(s[1].replace("Section", "")),
        )
        paras = []
        for s in sections:
            data = f.openstream(s).read()
            if compressed:
                data = zlib.decompress(data, -15)
            paras.extend(_section_text(data))
    return "\n".join(paras)


SPACED = re.compile(r"(?:[가-힣] ){2,}[가-힣]")  # '사 업 명'처럼 자간을 띄운 제목·셀
PDF_PAGE_NO = re.compile(r"[-–—]\s*\d{1,3}\s*[-–—]|\d{1,3}(?:\s*/\s*\d{1,3})?")


def _join_spaced(s: str) -> str:
    return s.replace(" ", "") if SPACED.fullmatch(s) else s


def _cell_text(text) -> str:
    return _join_spaced(" ".join((text or "").split()).replace("|", "/"))


def parse_pdf(path) -> str:
    """텍스트 PDF: 표는 find_tables()로 마크다운 복원, 나머지 글은 블록 단위로 위→아래 순서대로."""
    parts = []
    with pymupdf.open(path) as doc:
        for page in doc:
            tables = page.find_tables().tables
            boxes = [pymupdf.Rect(t.bbox) for t in tables]
            items = []
            for t in tables:
                rows = t.extract()
                cells = [(r, c, 1, 1, _cell_text(x)) for r, row in enumerate(rows) for c, x in enumerate(row)]
                items.append((t.bbox[1], _render_table(len(rows), t.col_count, cells)))
            for x0, y0, x1, y1, text, _, kind in page.get_text("blocks"):
                center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
                if kind == 0 and not PDF_PAGE_NO.fullmatch(text.strip()) \
                        and not any(center in box for box in boxes):  # 쪽번호·표 안 글(중복) 제외
                    items.append((y0, text))
            parts += [text for _, text in sorted(items, key=lambda it: it[0]) if text]
    return "\n".join(parts)


def extract_text(path):
    """HWP 5.x와 텍스트 PDF 지원. HWP 표는 마크다운 표로 복원. OCR은 추후 개선합니다."""
    path = Path(path)
    if path.suffix.lower() == ".hwp":
        try:
            text = parse_hwp(path)
        except (struct.error, zlib.error, KeyError, IndexError) as exc:
            raise ValueError(f"HWP 추출 실패: 암호화·손상·HWP3 형식인지 확인하세요. ({exc})") from exc
    elif path.suffix.lower() == ".pdf":
        text = parse_pdf(path)
    else:
        raise ValueError(f"지원하지 않는 형식: {path.suffix}")
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ue000-\uf8ff]", "", text)  # 제어문자 + PUA(깨진 기호)
    text = re.sub(r"[·.…ㆍ‥․]{5,}", " ", text)  # 목차 점선
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = "\n".join(_join_spaced(line) for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("추출된 텍스트 없음: 스캔 PDF는 OCR이 필요합니다.")
    return text


@observed("parse-documents")
def parse_documents(raw_dir="data/raw", output_dir="data/processed", limit=None):
    """반환: [{doc_id, text, metadata}]. limit은 CSV 앞쪽 N행입니다."""
    if limit is not None and limit < 1:
        raise ValueError("limit은 1 이상이어야 합니다.")
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    normalize = lambda name: unicodedata.normalize("NFC", name)
    files = {}
    for path in sorted((raw_dir / "files").iterdir()):
        if path.is_file():
            key = normalize(path.name)
            if key in files:
                raise ValueError(f"정규화 후 중복 파일명: {key}")
            files[key] = path
    with (raw_dir / "data_list.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "파일명" not in (reader.fieldnames or []):
            raise ValueError("CSV에 '파일명' 열이 필요합니다.")
        rows = list(reader)
    documents, errors, seen = [], [], set()
    for row in rows[:limit]:
        filename = normalize((row.get("파일명") or "").strip())
        try:
            if filename in seen:
                raise ValueError("CSV 중복 파일명: 메타데이터를 확인하세요.")
            seen.add(filename)
            path = files.get(filename)
            if path is None:
                raise ValueError("CSV 파일명에 해당하는 원본 파일 없음")
            text = extract_text(path)
            metadata = {k: v or "" for k, v in row.items() if k and k != "텍스트"}
            metadata.update(filename=filename, source=f"files/{filename}")
            documents.append({
                "doc_id": hashlib.sha256(filename.encode()).hexdigest()[:16],
                "text": text, "metadata": metadata,
            })
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append({"filename": filename, "error": str(exc)})
    write_json(output_dir / "documents.json", documents)
    write_json(output_dir / "parsing_errors.json", errors)
    if not documents:
        raise ValueError(f"추출 성공 문서가 없습니다. {output_dir}/parsing_errors.json 확인")
    return documents
