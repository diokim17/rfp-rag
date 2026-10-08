"""김연주: 팀 공통 평가셋(eval_team_v1)의 공통 지표로 검색 설정 비교 (답변 생성 없음).

  python experiments/team_eval_yjk.py --eval-file data/eval_team_v1.json --index-dir indexes/dev-20261007-structured
  python experiments/team_eval_yjk.py --eval-file ... --settings hybrid-ce-c50-cap2 --allow-api

- 공통 지표 (평가셋의 bm25_reference와 같은 이름)
  doc_recall@5           = 상위 5개 청크에 나온 정답 문서 수 / 정답 문서 수
  first_gold_rank        = 정답 문서 청크가 처음 나온 순위 (없으면 null)
  first_answer_chunk_rank = 정답 문서 청크 중 expected_keywords 하나 이상을 포함한 첫 청크의 순위 (없으면 null)
  순위는 retrieve가 돌려준 목록(top_k=50) 안에서만 셉니다. 리랭킹은 후보 50개만 재정렬하므로 50위 밖은 null입니다.
- 집계: 본 집계는 note가 있는 '기관명-필터표기' 문항(exact-match 필터가 0건)을 뺀 문항이며, 그 문항은 따로 보입니다.
- 질문 임베딩은 results/cache/query_embeddings.json 캐시를 쓰고, 캐시에 없을 때만 --allow-api로 호출합니다.
"""

import argparse
import hashlib
import math
import os
import statistics
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing import read_json, write_json  # noqa: E402
from retrieval_eval_yjk import CachedEmbeddingClient  # noqa: E402

TOP_K = 50
HYBRID = {"hybrid": True, "hybrid_vector_k": 100, "hybrid_bm25_k": 100}
# 하이브리드 후보(벡터 100 + BM25 100, RRF)는 yjk-0014와 같고, cross-encoder는 그중 상위 50개만 재정렬
SETTINGS = {
    "hybrid": {"rerank": "none", "candidates": 100, "max_per_doc": "none", **HYBRID},
    "hybrid-cap2": {"rerank": "none", "candidates": 100, "max_per_doc": 2, **HYBRID},
    "hybrid-ce-c50": {"rerank": "cross-encoder", "candidates": 50, "max_per_doc": "none", **HYBRID},
    "hybrid-ce-c50-cap2": {"rerank": "cross-encoder", "candidates": 50, "max_per_doc": 2, **HYBRID},
}
SEPARATE_TYPE = "기관명-필터표기"
GROUPS = ("쉬움", "보통", "어려움", "본 집계", SEPARATE_TYPE)


def normalize(text):
    return unicodedata.normalize("NFC", text).casefold()


def score_case(case, hits):
    """질문 하나의 공통 지표. hits는 retrieve 결과(순위순)."""
    gold = set(case["expected_doc_ids"])
    words = [normalize(word) for word in case.get("expected_keywords", [])]
    docs = [hit["doc_id"] for hit in hits]
    first_gold = next((rank for rank, doc in enumerate(docs, 1) if doc in gold), None)
    first_answer = next((rank for rank, hit in enumerate(hits, 1) if hit["doc_id"] in gold
                         and any(word in normalize(hit["text"]) for word in words)), None) if words else None
    return {"doc_recall@5": len(gold & set(docs[:5])) / len(gold), "first_gold_rank": first_gold,
            "first_answer_chunk_rank": first_answer, "returned": len(hits)}


def summarize(rows):
    """평균 doc_recall@5, 순위 지표는 1위 비율·5위 이내 비율·MRR(null은 0)."""
    out = {"cases": len(rows), "doc_recall@5": statistics.fmean(row["doc_recall@5"] for row in rows)}
    for key, short in (("first_gold_rank", "gold"), ("first_answer_chunk_rank", "answer")):
        ranks = [row[key] for row in rows]
        out[f"{short}_top1"] = statistics.fmean(r == 1 for r in ranks)
        out[f"{short}_top5"] = statistics.fmean(r is not None and r <= 5 for r in ranks)
        out[f"{short}_mrr"] = statistics.fmean(1 / r if r else 0.0 for r in ranks)
        out[f"{short}_missing"] = sum(r is None for r in ranks)
    return out


def group_of(case):
    return SEPARATE_TYPE if case.get("type") == SEPARATE_TYPE else "본 집계"


def summarize_groups(cases, rows):
    summary = {}
    for group in GROUPS:
        members = [row for case, row in zip(cases, rows)
                   if group == group_of(case) or (group_of(case) == "본 집계" and group == case["difficulty"])]
        if members:
            summary[group] = summarize(members)
    return summary


def bm25_rows(cases):
    """평가셋에 들어 있는 BM25 기준값을 같은 형식으로."""
    return [{"doc_recall@5": c["bm25_reference"]["bm25_doc_recall@5"],
             "first_gold_rank": c["bm25_reference"]["bm25_first_gold_rank"],
             "first_answer_chunk_rank": c["bm25_reference"]["bm25_first_answer_chunk_rank"]} for c in cases]


def file_hash(path):
    """평가셋·chunks.json 같은 JSON 파일 해시(줄바꿈을 LF로 맞춰 운영체제와 무관)."""
    from experiment_reports import text_file_hash
    return text_file_hash(path)


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def fmt_rank(rank):
    return "-" if rank is None else str(rank)


def report_lines(meta, cases, results):
    lines = ["# 팀 공통 평가셋 검색 지표 (eval_team_v1)", "",
             "검색 단계만 측정했습니다(답변 생성 없음). BM25 열은 평가셋에 들어 있는 bm25_reference 값이며 "
             "어느 인덱스·청킹으로 계산했는지는 평가셋에 적혀 있지 않습니다.", "",
             "| 항목 | 값 |", "| --- | --- |", *(f"| {k} | `{v}` |" for k, v in meta.items()), "",
             "지표: doc_recall@5 평균 / 정답 문서 첫 순위의 1위 비율·5위 이내 비율·MRR / "
             "정답 청크(정답 문서 + 기대 키워드 포함) 첫 순위의 1위 비율·5위 이내 비율·MRR. "
             f"순위는 상위 {TOP_K}개 안에서만 세며, 없으면 MRR 0으로 계산합니다.", ""]
    for group in GROUPS:
        rows = [(name, result["summary"].get(group)) for name, result in results]
        rows = [(name, row) for name, row in rows if row]
        if not rows:
            continue
        lines += [f"## {group} ({rows[0][1]['cases']}문항)", "",
                  "| 설정 | doc_recall@5 | 문서 1위 | 문서 5위 내 | 문서 MRR | 정답청크 1위 | 정답청크 5위 내 | 정답청크 MRR | 정답청크 없음 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for name, row in rows:
            lines.append(f"| {name} | {row['doc_recall@5']:.3f} | {row['gold_top1']:.3f} | {row['gold_top5']:.3f} | "
                         f"{row['gold_mrr']:.3f} | {row['answer_top1']:.3f} | {row['answer_top5']:.3f} | "
                         f"{row['answer_mrr']:.3f} | {row['answer_missing']} |")
        lines.append("")
    names = [name for name, _ in results]
    lines += ["## 문항별 (doc_recall@5 / 정답 문서 첫 순위 / 정답 청크 첫 순위)", "",
              "| id | 난이도 | 유형 | " + " | ".join(names) + " |", "| --- | --- | --- | " + " | ".join("---" for _ in names) + " |"]
    for i, case in enumerate(cases):
        cells = []
        for _, result in results:
            row = result["rows"][i]
            cells.append(f"{row['doc_recall@5']:.2f} / {fmt_rank(row['first_gold_rank'])} / {fmt_rank(row['first_answer_chunk_rank'])}")
        lines.append(f"| {case['id']} | {case['difficulty']} | {case['type']} | " + " | ".join(cells) + " |")
    lines += ["", "## 처리 시간과 적용 옵션", "", "| 설정 | 평균(ms) | 상위 5%(ms) | 적용 옵션 |", "| --- | ---: | ---: | --- |"]
    for name, result in results:
        if "options" in result:
            enabled = {k: v for k, v in result["options"].items() if v not in (None, False, "off", "none")}
            lines.append(f"| {name} | {result['mean_ms']:.1f} | {result['p95_ms']:.1f} | `{enabled}` |")
    lines += ["", "처리 시간은 retrieve() 한 번의 시간입니다. 질문 임베딩은 캐시를 썼고 BM25 색인 생성과 cross-encoder 모델 로드 시간은 제외했습니다.", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes/dev-20261007-structured")
    parser.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument("--cache-file", type=Path, default=ROOT / "results/cache/query_embeddings.json")
    parser.add_argument("--allow-api", action="store_true", help="캐시에 없는 질문 임베딩 API 호출을 허용")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from embedding import embed_texts, load_index
    from experiment_ids import next_experiment_id, owner_initials
    import retrieval

    cases = read_json(args.eval_file)
    index, chunks, config = load_index(args.index_dir)
    client = CachedEmbeddingClient(args.cache_file)
    model = config["embedding_model"]
    questions = sorted({case["question"] for case in cases})
    missing = [q for q in questions if hashlib.sha256(f"{model}\n{q}".encode()).hexdigest() not in client.cache]
    if missing and not args.allow_api:
        parser.error(f"캐시에 없는 질문 {len(missing)}건은 임베딩 API를 호출합니다. 확인 후 --allow-api를 붙이세요.")
    load_dotenv(ROOT / ".env", override=False)  # 키는 환경 변수로만 읽고 결과에 남기지 않습니다.
    experiment_id = next_experiment_id(owner_initials(os.getenv("EXPERIMENT_OWNER", "")), ROOT / ".experiment-state")
    print(f"실험 ID: {experiment_id} / 설정 {len(args.settings)}개 / 질문 {len(cases)}건 / 임베딩할 질문 {len(missing)}건")
    for start in range(0, len(missing), 64):
        embed_texts(missing[start:start + 64], client, model)
    client.save()

    start = time.perf_counter()
    retrieval._bm25_index(chunks)
    bm25_ms = (time.perf_counter() - start) * 1000
    ce_load_ms = 0.0
    if any(SETTINGS[name]["rerank"] == "cross-encoder" for name in args.settings):
        start = time.perf_counter()  # 모델 로드·GPU 초기화를 측정에서 분리
        retrieval.retrieve(cases[0]["question"], client, index, chunks, config, 1, None,
                           rerank="cross-encoder", candidates=1)
        ce_load_ms = (time.perf_counter() - start) * 1000

    results = [("BM25 기준값(평가셋)", {"rows": bm25_rows(cases)})]
    for name in args.settings:
        rows, times = [], []
        for case in cases:
            start = time.perf_counter()
            hits = retrieval.retrieve(case["question"], client, index, chunks, config, TOP_K, case.get("filters"),
                                      **SETTINGS[name])
            times.append((time.perf_counter() - start) * 1000)
            rows.append({"id": case["id"], **score_case(case, hits), "hits": [hit["chunk_id"] for hit in hits]})
        ordered = sorted(times)
        results.append((name, {"options": retrieval.retrieval_options(**SETTINGS[name]), "rows": rows,
                               "mean_ms": statistics.fmean(times),
                               "p95_ms": ordered[min(len(ordered) - 1, math.ceil(.95 * len(ordered)) - 1)]}))
        print(f"{name}: 완료 ({statistics.fmean(times):.0f}ms/질문)")
    for _, result in results:
        result["summary"] = summarize_groups(cases, result["rows"])
    if client.api_calls:
        client.save()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {"experiment_id": experiment_id, "measured_at_utc": stamp, "eval_file": relative(args.eval_file),
            "evaluation_sha256": file_hash(args.eval_file), "cases": len(cases), "index_dir": relative(args.index_dir),
            "chunks_sha256": file_hash(Path(args.index_dir) / "chunks.json"), "chunk_count": len(chunks),
            "chunking_strategy": config.get("chunking_strategy"), "embedding_model": model, "top_k": TOP_K,
            "bm25_build_ms": round(bm25_ms), "cross_encoder_model": os.getenv("RETRIEVAL_RERANK_MODEL", "BAAI/bge-reranker-v2-m3"),
            "cross_encoder_load_ms": round(ce_load_ms), "embedding_api_calls": client.api_calls}
    result_path = Path(args.results_dir) / f"team_eval_{experiment_id}_{stamp}.json"
    write_json(result_path, {"meta": meta, "results": dict(results)})
    report_path = Path(args.reports_dir) / f"team_eval_yjk_{experiment_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines(meta, cases, results)), encoding="utf-8")
    print(f"결과: {relative(result_path)}\n보고서: {relative(report_path)}")


if __name__ == "__main__":
    main()
