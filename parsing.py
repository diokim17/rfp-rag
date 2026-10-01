"""나상훈: 원본 파일 → Document 목록. CSV 열 이름은 metadata에 유지합니다."""

import csv
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_text(path):
    """HWP 5.x와 텍스트 PDF 지원. OCR·표 구조 복원은 추후 개선합니다."""
    path = Path(path)
    if path.suffix.lower() == ".hwp":
        result = subprocess.run(
            [sys.executable, "-c", "from hwp5.hwp5txt import main; main()", str(path)],
            capture_output=True, timeout=120,
        )
        if result.returncode:
            raise ValueError("HWP 추출 실패: 암호화·손상·지원 형식을 확인하세요.")
        text = result.stdout.decode("utf-8")
    elif path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        raise ValueError(f"지원하지 않는 형식: {path.suffix}")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("추출된 텍스트 없음: 스캔 PDF는 OCR이 필요합니다.")
    return text


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
        except (ValueError, OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
            errors.append({"filename": filename, "error": str(exc)})
    write_json(output_dir / "documents.json", documents)
    write_json(output_dir / "parsing_errors.json", errors)
    if not documents:
        raise ValueError(f"추출 성공 문서가 없습니다. {output_dir}/parsing_errors.json 확인")
    return documents
