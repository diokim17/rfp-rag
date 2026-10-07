"""나상훈: 원본 파일 → Document 목록. CSV 열 이름은 metadata에 유지합니다."""

import bisect
import csv
import hashlib
import itertools
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


TABLE_OPEN, TABLE_CLOSE = "<!-- table -->", "<!-- /table -->"


def _number_tables(text: str) -> str:
    """표 시작·끝 표시에 문서 안 순번 ID를 붙임: <!-- table:T1 --> ... <!-- /table:T1 -->"""
    ids = itertools.count(1)
    return re.sub(f"{TABLE_OPEN}(.*?){TABLE_CLOSE}",
                  lambda m: f"<!-- table:T{(i := next(ids))} -->{m[1]}<!-- /table:T{i} -->",
                  text, flags=re.S)


def _render_table(rows: int, cols: int, cells: list, nested: bool = False) -> str:
    """cells [(row, col, rowspan, colspan, text)] → 마크다운 표.
    세로 병합 셀은 값을 반복해서 행 하나만 떼어 봐도 뜻이 통하게 함 (청킹 대비)."""
    grid = [[""] * cols for _ in range(rows)]
    filled = {(r, c) for r, c, _, _, text in cells if text}
    for r, c, rs, cs, text in cells:
        # 다단 헤더: 바로 아래 행이 이 범위에 글자 있는 칸을 2개 이상 두면 상위 헤더로 보고 반복.
        # 첫 열 병합은 행 제목·각주, 표 전체 폭은 제목, 3행 이하는 중간 구분 행이라 제외
        # ponytail: 위치 기반 휴리스틱. 3단 이상 헤더가 많이 보이면 r 상한을 올릴 것
        is_parent = r < 2 and 0 < c and cs < cols and sum((r + rs, cc) in filled for cc in range(c, c + cs)) >= 2
        span = cs if is_parent else 1
        for rr in range(r, min(r + rs, rows)):
            for cc in range(c, min(c + span, cols)):
                grid[rr][cc] = text
    lines = [row for row in grid if any(row)]
    lines = [row for i, row in enumerate(lines) if i == 0 or row != lines[i - 1]]  # 병합 반복으로 똑같아진 행
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
    return "\n".join([TABLE_OPEN, *out, TABLE_CLOSE])  # 번호는 extract_text에서 문서 단위로


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
PDF_FRAME_HEAD = re.compile(r"페\s*이\s*지\s*:\s*\d+\s*/\s*\d+")  # 쪽 테두리 표 머리말('페 이 지 : 4/19')


def _join_spaced(s: str) -> str:
    return s.replace(" ", "") if SPACED.fullmatch(s) else s


def _cell_text(text) -> str:
    return _join_spaced(" ".join((text or "").split()).replace("|", "/"))


def _reading_order(items, width):
    """좌우 2단(두 쪽 모아찍기 등) 페이지면 왼쪽 단 → 오른쪽 단, 아니면 위 → 아래."""
    mid = width / 2
    left = sum(x1 <= mid for x0, x1, *_ in items)
    right = sum(x0 >= mid for x0, x1, *_ in items)
    if left >= 3 and right >= 3 and len(items) - left - right <= 2:
        return lambda it: (it[0] >= mid, it[2])
    return lambda it: it[2]


def _merge_pages(pages):
    """쪽별 [글(str) | 표(행 list)] → 한 목록.
    쪽 끝 표와 다음 쪽 첫 표가 같은 표면 하나로 이음 (반복된 헤더 행은 한 번만)."""
    out = []
    for page in pages:
        if page and out and isinstance(page[0], list) and isinstance(out[-1], list):
            a, b = out[-1], page[0]
            shared = {x for x in a[0] if x} & {x for x in b[0] if x}
            # 첫 행이 같으면 헤더 반복, 하나도 안 겹치면 헤더 없이 이어짐. 일부만 같으면 별개 표(요구사항 카드 등)
            # ponytail: 열 수가 같고 헤더가 안 겹치는 별개 표도 이어 붙음. 오탐이 보이면 쪽 위치 조건 추가
            if len(a[0]) == len(b[0]) and (b[0] == a[0] or not shared):
                a += b[1:] if b[0] == a[0] else b
                page = page[1:]
        out += page
    return out


def parse_pdf(path) -> str:
    """텍스트 PDF: 표는 find_tables()로 마크다운 복원, 나머지 글은 블록 단위로 위→아래 순서대로."""
    pages = []
    with pymupdf.open(path) as doc:
        for page in doc:
            tables, heads = [], []
            for t in page.find_tables().tables:
                rows = t.extract()
                # 본문 전체를 감싼 쪽 테두리 표: 표로 보지 않고 머리말 행만 버림 (안쪽 글은 일반 블록으로)
                if rows and PDF_FRAME_HEAD.search(" ".join(c or "" for c in rows[0])):
                    heads.append(pymupdf.Rect(t.rows[0].bbox))
                else:
                    tables.append((t, rows))
            boxes = [pymupdf.Rect(t.bbox) for t, _ in tables]
            items = [(t.bbox[0], t.bbox[2], t.bbox[1], rows) for t, rows in tables]  # (x0, x1, y0, 글|표)
            for x0, y0, x1, y1, text, _, kind in page.get_text("blocks"):
                center = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
                if kind == 0 and text.strip() and not PDF_PAGE_NO.fullmatch(text.strip()) \
                        and not any(center in box for box in boxes + heads):  # 쪽번호·표 안 글(중복)·테두리 머리말 제외
                    items.append((x0, x1, y0, text))
            pages.append([it[3] for it in sorted(items, key=_reading_order(items, page.rect.width)) if it[3]])
    parts = []
    for x in _merge_pages(pages):
        if isinstance(x, list):
            x = _render_table(len(x), len(x[0]), [(r, c, 1, 1, _cell_text(v))
                                                  for r, row in enumerate(x) for c, v in enumerate(row)])
        if x:
            parts.append(x)
    return "\n".join(parts)


def _clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ue000-\uf8ff]", "", text)  # 제어문자 + PUA(깨진 기호)
    text = re.sub(r"[·.…ㆍ‥․]{5,}", " ", text)  # 목차 점선
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\(?이 ?하 ?여 ?백\)?\n?", "", text)  # 서식 끝 표시
    text = re.sub(r"\[\n+\]", "[ ]", text)  # PDF에서 줄이 갈린 체크박스
    text = re.sub(r"□\n+√", "☑", text)
    text = re.sub(r"Ÿ\s*", "• ", text)  # PDF 깨진 글머리표(Wingdings)
    text = re.sub(r"(^|\s)\u00ad ?", r"\1• ", text, flags=re.M)  # 글머리표로 쓰인 soft hyphen
    text = text.replace("\u00ad", "-")  # 단어 중간(전화번호 등)은 하이픈
    text = "\n".join(_join_spaced(line) for line in text.split("\n"))
    # '사 업 비: 금…'처럼 콜론 앞 2~4글자 라벨의 자간 공백 제거 (앞에 띄운 글자가 더 있으면 서식 칸이라 둠)
    text = re.sub(r"(?<![가-힣])(?<![가-힣] )((?:[가-힣] ){1,3}[가-힣])(?=\s*[:：])", lambda m: m[1].replace(" ", ""), text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


TOC_ITEM = r"(?:[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]\.?|\d{1,2}(?:\.\d{1,2})*\.?|[가-하]\.|제\s?\d+\s?장|[\[<][가-힣\s]{1,6}\d{1,2}[\]>])\s?"  # [양식 1]·<붙임 2> 등 붙임 목록 포함
TOC_LINES = re.compile(rf"(?:^{TOC_ITEM}[^\n]{{1,40}}?\s\d{{1,3}}\n){{3,}}", re.M)  # 제목 + 쪽번호 줄 3개 이상
TOC_INLINE = re.compile(rf"{TOC_ITEM}[가-힣][가-힣 ·]{{1,20}}\s\d{{1,3}}\s")  # 표 한 칸·한 문단에 몰아 쓴 목차


def _drop_toc(text: str) -> str:
    """앞부분(20%) 목차 제거: 제목이 다 모여 있어 어떤 질문에도 검색돼 top-k를 차지함. 섹션 정보는 sections로 대체.
    본문 뒤쪽의 '1. 서버 2' 같은 목록은 건드리지 않게 위치로 제한."""
    head = len(text) // 5
    text = TOC_LINES.sub(lambda m: "" if m.start() < head else m[0], text)
    many = lambda m: "" if m.start() < head and len(TOC_INLINE.findall(m[0] + " ")) >= 5 else m[0]  # 한 덩어리에 목차 항목 5개+
    text = re.sub(f"{TABLE_OPEN}.*?{TABLE_CLOSE}\n?", many, text, flags=re.S)
    text = re.sub(r"^[^|<\n].*\n?", many, text, flags=re.M)  # 한 문단으로 이어 쓴 목차
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"^목\s*차\n", "", text, flags=re.M))


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
    text = _number_tables(_drop_toc(_clean_text(text)))
    if not text:
        raise ValueError("추출된 텍스트 없음: 스캔 PDF는 OCR이 필요합니다.")
    return text


ROMAN, KOR = "ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ", "가나다라마바사아자차카타파하"
HEADINGS = (  # (단계, 정규식): 장 Ⅰ / 제1장 → 절 1. → 1.1 → 1.1.1 → 항 가.
    (0, re.compile(rf"([{ROMAN}])\.?\s*[가-힣].*")),
    (0, re.compile(r"제\s?(\d{1,2})\s?장\s*[가-힣].*")),
    (1, re.compile(r"(\d{1,2})\.?\s+[가-힣][^:：]*")),
    (2, re.compile(r"\d{1,2}\.(\d{1,2})\.?\s+[가-힣][^:：]*")),
    (3, re.compile(r"\d{1,2}\.\d{1,2}\.(\d{1,2})\.?\s+[가-힣][^:：]*")),
    (4, re.compile(rf"([{KOR}])\.\s*[가-힣][^:：]*")),
)
DEPTH = 5


def _sections(text: str) -> list:
    """[[시작 위치, "Ⅳ 제안요청 내용 > 2. 상세 요구사항"], ...]. 30자 이하·번호가 1 또는 직전+1인 줄만 제목으로 봄."""
    out, path, last = [], [None] * DEPTH, [0] * DEPTH
    for m in re.finditer(r"^.+$", text, re.M):
        line = m[0].strip()
        if len(line) > 30 or line[:1] in "|<" or re.search(r"\s\d+$", line):  # 긴 문장·표·목차(쪽번호) 제외
            continue
        for level, pattern in HEADINGS:
            h = pattern.fullmatch(line)
            if not h:
                continue
            n = int(h[1]) if h[1].isdigit() else (ROMAN if level == 0 else KOR).find(h[1]) + 1
            if n in (1, last[level] + 1) or (level == 0 and n > last[level]):  # 장 번호는 빠진 장이 있어도 허용
                if n == 1 and last[level]:  # 상위 제목 없이 번호가 다시 시작 → 놓친 상위 제목, 틀린 경로보다 빈 경로
                    path[:level] = [None] * level
                last[level:] = [n] + [0] * (DEPTH - 1 - level)
                path[level:] = [line] + [None] * (DEPTH - 1 - level)
                out.append([m.start(), " > ".join(p for p in path if p)])
                break
    return out


def section_of(document, pos: int) -> str:
    """청킹용: 원문 위치(start_char)가 속한 섹션 경로. 첫 제목 앞이면 빈 문자열."""
    sections = document.get("sections") or []
    i = bisect.bisect_right([s for s, _ in sections], pos)
    return sections[i - 1][1] if i else ""


WON_UNIT = {"억": 10**8, "천만": 10**7, "백만": 10**6, "만": 10**4, "천": 10**3}


def _won(s: str):
    """'130,000,000원', '40,000천원', '196백만원', '1억 5천만 원' → 원 단위 int. 금액이 없으면 None.
    '원'이 없어도 '￦ 60,000,000', '50,000,000(VAT포함)'처럼 ￦ 뒤 숫자나 쉼표로 묶인 백만 이상 숫자는 원으로 봄."""
    s = s.replace(" ", "")
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)(억|천만|백만|만|천)?(?:(\d[\d,]*)(천만|백만|만))?원", s)
    if not m:
        m = re.search(r"[￦₩](\d[\d,]*)|(?<![\d,.])(\d{1,3}(?:,\d{3}){2,})(?![\d,])", s)
        return int((m[1] or m[2]).replace(",", "")) if m else None
    value = float(m[1].replace(",", "")) * WON_UNIT.get(m[2], 1)
    if m[3]:
        value += float(m[3].replace(",", "")) * WON_UNIT[m[4]]
    return int(value)


_labels = lambda *names: "|".join(r"\s?".join(name) for name in names)  # '사업 예산'처럼 띄어 써도 매칭
FIELDS = {  # metadata 키: (라벨 묶음들, 값 확인). 앞 묶음(구체적 라벨)에서 못 찾을 때만 뒤 묶음(범용 라벨)
    "원문 사업 예산": ((_labels("사업예산액", "사업예산", "소요예산", "총사업비", "사업금액", "사업비", "배정예산",
                          "기초금액", "추정가격", "예산소요액", "사업규모", "과업예산", "용역예산", "용역금액",
                          "용역비용", "설계금액", "집행한도액"), _labels("예산액", "예산")),
                 lambda v: _won(v) and not re.search("미만|이상|초과", v)),
    "원문 사업 기간": ((_labels("사업수행기간", "과업수행기간", "사업기간", "계약기간", "용역기간", "과업기간", "구축기간"), _labels("기간")),
                 lambda v: re.search(r"\d+\s*(?:일|개월|년|월)|\d{2,4}\s*[.\-]\s*\d|계약일|착수일|체결일", v)),
    "원문 계약 방법": ((_labels("입찰및계약방법", "낙찰자결정방법", "계약방법", "계약방식", "입찰방식", "입찰방법",
                          "낙찰방식", "사업추진방식"),),
                 lambda v: re.search("경쟁|협상|수의|입찰", v)),
}


def _fields(text: str) -> dict:
    """원문에서 예산·기간·계약 방법 추출. CSV 값 검수용 (CSV 사업 금액은 대부분 VAT 포함 금액).
    라벨 뒤 ':' / ')' / 표 칸 '|' 다음 값 중 확인을 통과한 첫 번째. 없으면 다음 줄 글머리('- ', 'ㅇ ') 뒤 값."""
    values = lambda labels, sep: (m[1].strip()[:100]
                                  for m in re.finditer(rf"(?<![가-힣])(?:{labels}){sep}\s*([^|\n]+)", text))
    seps = (r"\s*[:：)|]", r"[ \t]*\n[-–ㅇ○□•]")
    out = {key: next((v for labels in groups for sep in seps for v in values(labels, sep) if ok(v)), "")
           for key, (groups, ok) in FIELDS.items()}
    out["원문 사업 금액"] = str(_won(out["원문 사업 예산"]) or "")
    return out


@observed("parse-documents")
def parse_documents(raw_dir="data/raw", output_dir="data/processed", limit=None):
    """반환: [{doc_id, text, metadata, sections}]. limit은 CSV 앞쪽 N행입니다.
    metadata: CSV 열 + 원문에서 뽑은 '원문 사업 예산'·'원문 사업 금액'(원, 숫자 문자열)·'원문 사업 기간'·'원문 계약 방법'
    (못 찾으면 빈 문자열). sections는 metadata와 달리 청크마다 복사되지 않으니 section_of()로 조회하세요."""
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
            metadata.update(filename=filename, source=f"files/{filename}", **_fields(text))
            documents.append({
                "doc_id": hashlib.sha256(filename.encode()).hexdigest()[:16],
                "text": text, "metadata": metadata, "sections": _sections(text),
            })
        except (ValueError, OSError, UnicodeError) as exc:
            errors.append({"filename": filename, "error": str(exc)})
    write_json(output_dir / "documents.json", documents)
    write_json(output_dir / "parsing_errors.json", errors)
    if not documents:
        raise ValueError(f"추출 성공 문서가 없습니다. {output_dir}/parsing_errors.json 확인")
    return documents
