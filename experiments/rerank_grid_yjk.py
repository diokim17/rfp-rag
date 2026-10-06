"""김연주: 리랭킹 방식 × 후보 수 × top-k 조합별 Recall@k·MRR@k와 처리 시간 비교 (생성 API 호출 없음).

  python experiments/rerank_grid_yjk.py --eval-file data/eval_sample_v0.json --allow-api

실행 순서: 리랭킹(lexical → cross-encoder) → 후보 수(30, 50, 70, 100) → top-k(3, 5, 10).
비교 기준으로 리랭킹 없는 코사인 검색(none)도 top-k마다 먼저 측정합니다.

지표 (평가 담당 evaluation.py와 같은 정의, 문서 단위):
  Recall@k = 상위 k개 청크에 나온 정답 문서 수 / 정답 문서 수
  MRR@k    = 상위 k개 안에서 처음 나온 정답 문서 순위의 역수 (없으면 0)
처리 시간: 질문 임베딩은 측정 전에 캐시에 채워 두므로 검색·리랭킹 시간만 잽니다.
cross-encoder 모델 로드 시간은 첫 질문 지연에 섞이지 않도록 따로 기록합니다.
"""

import argparse
import hashlib
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parsing import read_json, write_json  # noqa: E402
from retrieval_eval_yjk import CachedEmbeddingClient  # noqa: E402

MODES = ("lexical", "cross-encoder")
CANDIDATES = (30, 50, 70, 100)
TOP_KS = (3, 5, 10)
SEED_CACHE = ROOT / "results/cache/query_embeddings.json"


def score(case, hits, top_k):
    """질문 하나의 Recall@k, MRR@k."""
    gold = set(case["expected_doc_ids"])
    docs = [hit["doc_id"] for hit in hits[:top_k]]
    first = next((rank for rank, doc in enumerate(docs, 1) if doc in gold), None)
    return {"recall": len(gold & set(docs)) / len(gold), "mrr": 1 / first if first else 0.0}


def run_setting(cases, search, top_k):
    """질문마다 검색 시간을 재고 평균 지표와 시간 통계를 반환합니다."""
    rows, times = [], []
    for case in cases:
        start = time.perf_counter()
        hits = search(case["question"], case.get("filters"))
        times.append((time.perf_counter() - start) * 1000)
        rows.append({"id": case.get("id"), **score(case, hits, top_k),
                     "hits": [hit["doc_id"] for hit in hits[:top_k]]})
    ordered = sorted(times)
    return {
        "recall": statistics.fmean(row["recall"] for row in rows),
        "mrr": statistics.fmean(row["mrr"] for row in rows),
        "total_ms": sum(times), "mean_ms": statistics.fmean(times), "median_ms": statistics.median(times),
        "p95_ms": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))], "max_ms": ordered[-1],
        "misses": [row["id"] for row in rows if row["recall"] < 1],
    }, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes/parsing-v2-yjk")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    parser.add_argument("--cache-file", type=Path, default=SEED_CACHE)
    parser.add_argument("--allow-api", action="store_true", help="캐시에 없는 질문의 임베딩 API 호출을 허용")
    args = parser.parse_args()

    cases = read_json(args.eval_file)
    if not isinstance(cases, list) or not cases:
        parser.error("평가 파일에 질문 목록이 없습니다.")
    for number, case in enumerate(cases):
        if not case.get("question") or not case.get("expected_doc_ids"):
            parser.error(f"{number}번 질문에 question 또는 expected_doc_ids가 없습니다.")

    from embedding import embed_texts, load_index
    from retrieval import retrieve

    index, chunks, config = load_index(args.index_dir)
    model = config["embedding_model"]
    client = CachedEmbeddingClient(args.cache_file)
    missing = sorted({case["question"] for case in cases
                      if hashlib.sha256(f"{model}\n{case['question']}".encode()).hexdigest() not in client.cache})
    print(f"질문 {len(cases)}건 / 캐시에 없는 질문 {len(missing)}개")
    if missing and not args.allow_api:
        sys.exit("캐시에 없는 질문이 있습니다. 임베딩 API 호출을 허용하려면 --allow-api를 추가하세요.")
    embed_ms = 0.0
    if missing:
        start = time.perf_counter()
        for begin in range(0, len(missing), 64):
            embed_texts(missing[begin:begin + 64], client, model)
        embed_ms = (time.perf_counter() - start) * 1000
        client.save()
        print(f"질문 임베딩: {len(missing)}개, {embed_ms:.0f}ms (API {client.api_calls}회)")
    api_calls_before = client.api_calls

    # 준비: cross-encoder 모델 로드와 GPU 초기화를 측정에서 분리합니다.
    start = time.perf_counter()
    retrieve(cases[0]["question"], client, index, chunks, config, 1, None, rerank="cross-encoder", candidates=1)
    load_ms = (time.perf_counter() - start) * 1000
    print(f"cross-encoder 모델 준비: {load_ms:.0f}ms\n")

    settings = [("none", None, k) for k in TOP_KS]
    settings += [(mode, candidates, k) for mode in MODES for candidates in CANDIDATES for k in TOP_KS]
    summary, records = [], []
    print("| 리랭킹 | 후보 수 | top-k | Recall@k | MRR@k | 질문당 평균(ms) | 중앙(ms) | p95(ms) | 전체(ms) |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mode, candidates, top_k in settings:
        def search(question, filters, mode=mode, candidates=candidates, top_k=top_k):
            return retrieve(question, client, index, chunks, config, top_k, filters,
                            rerank=mode, candidates=candidates)

        result, rows = run_setting(cases, search, top_k)
        summary.append({"rerank": mode, "candidates": candidates, "top_k": top_k, **result})
        records += [{"rerank": mode, "candidates": candidates, "top_k": top_k, **row} for row in rows]
        print(f"| {mode} | {candidates or '-'} | {top_k} | {result['recall']:.3f} | {result['mrr']:.3f} | "
              f"{result['mean_ms']:.1f} | {result['median_ms']:.1f} | {result['p95_ms']:.1f} | {result['total_ms']:.0f} |")
    if client.api_calls != api_calls_before:
        sys.exit("측정 중에 임베딩 API가 호출되었습니다. 캐시 키 규칙을 확인하세요.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {**config, "index_dir": str(args.index_dir), "eval_file": args.eval_file.name, "cases": len(cases),
            "evaluation_sha256": hashlib.sha256(args.eval_file.read_bytes()).hexdigest(),
            "query_embedding_api_calls": client.api_calls, "query_embedding_ms": round(embed_ms),
            "cross_encoder_load_ms": round(load_ms), "measured_at_utc": stamp}
    output = args.results_dir / f"rerank_grid_{stamp}.json"
    write_json(output, {"settings": meta, "summary": summary, "records": records})

    # 공유용 요약: 질문·원문·필터 값 없이 집계 지표만 기록
    lines = [f"# 리랭킹 × 후보 수 × top-k 비교: {args.eval_file.stem}", "", "| 설정 | 값 |", "| --- | --- |"]
    lines += [f"| {key} | `{value}` |" for key, value in meta.items()]
    lines += ["", "| 리랭킹 | 후보 수 | top-k | Recall@k | MRR@k | 질문당 평균(ms) | 중앙(ms) | p95(ms) | 전체(ms) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines += [f"| {row['rerank']} | {row['candidates'] or '-'} | {row['top_k']} | {row['recall']:.3f} | {row['mrr']:.3f} | "
              f"{row['mean_ms']:.1f} | {row['median_ms']:.1f} | {row['p95_ms']:.1f} | {row['total_ms']:.0f} |"
              for row in summary]
    lines += ["", "Recall@k·MRR@k는 문서 단위 정답 기준입니다. 처리 시간은 질문 임베딩 캐시를 사용한 검색·리랭킹 시간이며 "
              "임베딩 API 지연과 cross-encoder 모델 로드 시간은 제외했습니다.", ""]
    report = args.reports_dir / f"rerank_grid_yjk_{args.eval_file.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장: {output}\n공유용 요약: {report}")


if __name__ == "__main__":
    main()
