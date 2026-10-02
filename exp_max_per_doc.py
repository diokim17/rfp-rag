"""검색 단계만 평가하는 실험 스크립트 (답변 생성 호출 없음, run.py는 수정하지 않음).

사용 예:
    python exp_max_per_doc.py --eval-file data/eval_retrieval_yjk_e2e24.json --max-per-doc 2 3

기준선(문서당 상한 없음)은 항상 먼저 실행되고, --max-per-doc 값마다 같은 평가셋으로 비교합니다.
lexical 등 다른 검색 옵션을 함께 쓰려면 아래 retrieve 호출 부분에 인자를 추가하세요.
"""
import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# TODO: 평가 JSON의 실제 키 이름에 맞게 수정하세요.
QUESTION_KEY = "question"
GOLD_KEY = "expected_doc_ids"        # 정답 문서 ID (여러 개면 리스트도 가능)
FILTER_KEY = "filters"     # 질문별 필터가 있는 경우


def evaluate_cases(cases, search, top_k):
    """search(question, filters) -> 점수순 hits 리스트(각 hit에 doc_id)."""
    found, rr_sum, times, misses = 0, 0.0, [], []
    for number, case in enumerate(cases):
        start = time.perf_counter()
        hits = search(case[QUESTION_KEY], case.get(FILTER_KEY) or {})
        times.append((time.perf_counter() - start) * 1000)
        ids = [hit["doc_id"] for hit in hits[:top_k]]
        gold = case[GOLD_KEY]
        gold = set(gold) if isinstance(gold, (list, tuple, set)) else {gold}
        rank = next((i + 1 for i, doc_id in enumerate(ids) if doc_id in gold), None)
        if rank:
            found += 1
            rr_sum += 1 / rank
        else:
            misses.append(number)  # 놓친 질문 번호 (원인 분석용)
    n = len(cases)
    return {"recall": found / n, "mrr": rr_sum / n, "median_ms": statistics.median(times),
            "cases": n, "misses": misses}


def main():
    parser = argparse.ArgumentParser(description="검색 단계 실험")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-doc", type=int, nargs="*", default=[],
                        help="비교할 문서당 최대 청크 수 (기준선은 항상 포함)")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from openai import OpenAI
    from embedding import load_index
    from retrieval import retrieve

    load_dotenv(ROOT / ".env", override=False)  # 키는 이 컴퓨터의 .env에서만 읽습니다.
    client = OpenAI(timeout=60.0, max_retries=2)
    index, chunks, config = load_index(args.index_dir)
    cases = json.loads(args.eval_file.read_text(encoding="utf-8"))
    if isinstance(cases, dict):  # TODO: 평가 파일이 {"cases": [...]} 같은 구조라면 키 이름 확인
        cases = cases.get("cases") or cases.get("questions")

    results = {}
    for limit in [None, *args.max_per_doc]:
        def search(question, filters, limit=limit):
            extra = {} if limit is None else {"max_per_doc": limit}
            return retrieve(question, client, index, chunks, config, args.top_k, filters, **extra)

        name = "기준선" if limit is None else f"문서당 최대 {limit}개"
        result = evaluate_cases(cases, search, args.top_k)
        results[name] = result
        print(f"{name}: Recall@{args.top_k} {result['recall']:.3f} / MRR {result['mrr']:.3f} / "
              f"중앙 지연 {result['median_ms']:.0f}ms / 놓친 {len(result['misses'])}건")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.results_dir / f"retrieval_exp_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"eval_file": str(args.eval_file), "top_k": args.top_k,
                                  "results": results}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"결과 저장: {output}")


if __name__ == "__main__":
    main()
