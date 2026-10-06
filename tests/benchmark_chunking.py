"""API/.env 접근 없이 실제 문서의 청킹 계약·품질·시간을 비교하는 수동 검증.

프로젝트 루트에서 python tests/benchmark_chunking.py --baseline-ref <commit> 실행.
원본은 읽기만 하며 복사본/청크/집계를 타임스탬프별 비공개 경로에 저장합니다.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embedding import chunk_documents, _embedding_input
from parsing import read_json, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(documents, chunks, size, overlap):
    grouped = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["doc_id"]].append(chunk)
    assert len({d["doc_id"] for d in documents}) == len(documents)
    assert set(grouped) <= {d["doc_id"] for d in documents}
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    lengths = sorted(len(c["text"]) for c in chunks)
    result = dict(chunks=len(chunks), embedding_requests=(len(chunks) + 31) // 32,
                  length_min=min(lengths), length_median=statistics.median(lengths),
                  length_p95=lengths[(95 * len(lengths) + 99) // 100 - 1], length_max=max(lengths),
                  short_under_150=sum(n < 150 for n in lengths), overlap_chars=0,
                  small_tables=0, small_tables_not_whole=0, small_table_cuts=0,
                  table_interior_boundaries=0, table_midrow_boundaries=0,
                  table_midrow_boundaries_in_short_rows=0,
                  chunks_crossing_section=0, document_chars=sum(len(d["text"]) for d in documents))
    # 번호가 붙은 실제 표를 구현의 경계 검출과 독립적으로 집계합니다.
    tables = re.compile(r"<!-- table:(T\d+) -->.*?<!-- /table:\1 -->", re.S)
    for doc in documents:
        text, items = doc["text"], grouped[doc["doc_id"]]
        coverage = bytearray(len(text))
        last_start, last_end = -1, 0
        boundaries = set()
        for c in items:
            start, end = (c["metadata"][k] for k in ("start_char", "end_char"))
            assert 0 <= start < end <= len(text) and end - start <= size
            assert start > last_start and end > last_end
            assert max(0, last_end - start) <= overlap
            assert c["text"] == text[start:end] and c["text"].strip()
            assert c["chunk_id"].startswith(doc["doc_id"] + ":")
            assert all(c["metadata"][k] == v for k, v in doc["metadata"].items())
            result["overlap_chars"] += max(0, last_end - start)
            coverage[start:end] = b"\1" * (end - start)
            boundaries.update((start, end))
            last_start, last_end = start, end
        assert all(coverage[i] for i, char in enumerate(text) if not char.isspace())
        assert json.loads(json.dumps(items)) == items
        boundaries = sorted(boundaries)
        for m in tables.finditer(text):
            interior = boundaries[bisect_right(boundaries, m.start()):bisect_left(boundaries, m.end())]
            result["table_interior_boundaries"] += len(interior)
            result["table_midrow_boundaries"] += sum(text[pos - 1] != "\n" for pos in interior)
            row_start = m.start()
            for line in text[m.start():m.end()].splitlines(keepends=True):
                row_end = row_start + len(line)
                if len(line) <= size:
                    result["table_midrow_boundaries_in_short_rows"] += sum(
                        row_start < pos < row_end and text[pos - 1] != "\n" for pos in interior)
                row_start = row_end
            if m.end() - m.start() <= size:
                result["small_tables"] += 1
                result["small_table_cuts"] += len(interior)
                result["small_tables_not_whole"] += not any(
                    c["metadata"]["start_char"] <= m.start() and c["metadata"]["end_char"] >= m.end()
                    for c in items)
        sections = [s for s, _ in doc.get("sections", [])]
        result["chunks_crossing_section"] += sum(
            bisect_left(sections, c["metadata"]["end_char"]) > bisect_right(sections, c["metadata"]["start_char"])
            for c in items)
    result["context_extra_chars"] = sum(len(_embedding_input(c, "project")) - len(c["text"]) for c in chunks)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--baseline-ref", required=True, help="기존 fixed/boundary 동작을 검증할 로컬 Git 커밋")
    args = parser.parse_args()
    original_files = [args.source_processed_dir / "documents.json"] + sorted((ROOT / "indexes").glob("*"))
    originals = {p: digest(p) for p in original_files if p.is_file()}
    label = "chy-project-context-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    processed, output = ROOT / "data/processed" / label, ROOT / "results" / label
    processed.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.source_processed_dir / "documents.json", processed / "documents.json")
    errors = args.source_processed_dir / "parsing_errors.json"
    if errors.exists():
        shutil.copyfile(errors, processed / errors.name)
    documents = read_json(processed / "documents.json")
    before = json.dumps(documents, ensure_ascii=False)
    baseline = ModuleType("baseline_embedding")
    code = subprocess.check_output(["git", "show", f"{args.baseline_ref}:embedding.py"], cwd=ROOT, text=True)
    exec(compile(code, "baseline_embedding.py", "exec"), baseline.__dict__)
    result = {"artifact_label": label, "document_count": len(documents), "chunk_size": 1000,
              "chunk_overlap": 150, "baseline_ref": args.baseline_ref,
              "python": platform.python_version(), "platform": platform.platform(),
              "documents_sha256": digest(processed / "documents.json"),
              "embedding_py_sha256": digest(ROOT / "embedding.py"),
              "benchmark_sha256": digest(Path(__file__)), "api_calls": 0, "strategies": {}}
    result["baseline_index_hashes"] = {p.name: digest(p) for p in (ROOT / "indexes").glob("*") if p.is_file()}
    result["evaluation_hashes"] = {p.name: digest(p) for p in (ROOT / "data").glob("eval*_v0.json")}
    samples = {s: [] for s in ("fixed", "boundary", "structured")}
    for strategy in samples:
        with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
            chunks = chunk_documents(documents)  # 준비 실행 및 결과 계약 검사
            metrics = measure(documents, chunks, 1000, 150)
            if strategy != "structured":
                assert chunks == baseline.chunk_documents(documents)
                metrics["baseline_chunks_identical"] = True
            else:
                assert metrics["small_tables_not_whole"] == metrics["small_table_cuts"] == 0
                assert metrics["table_midrow_boundaries_in_short_rows"] == 0
            write_json(output / strategy / "chunks.json", chunks)
            metrics["chunks_sha256"] = digest(output / strategy / "chunks.json")
            result["strategies"][strategy] = metrics
    for iteration in range(5):
        order = list(samples) if iteration % 2 == 0 else list(reversed(samples))
        for strategy in order:
            with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
                start = time.perf_counter()
                chunk_documents(documents)
                samples[strategy].append((time.perf_counter() - start) * 1000)
    for strategy, times in samples.items():
        result["strategies"][strategy]["chunking_ms_samples"] = times
        result["strategies"][strategy]["chunking_ms_median"] = statistics.median(times)
    assert before == json.dumps(documents, ensure_ascii=False)
    assert all(digest(p) == value for p, value in originals.items())
    result["source_and_baseline_unchanged"] = True
    write_json(output / "summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
