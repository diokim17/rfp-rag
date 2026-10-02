"""김연주: 질문 임베딩 → FAISS 검색 → (선택) 리랭킹 → 문서당 청크 상한 적용 → 검색 결과(hits) 반환.

검색 단계만 평가하는 실험 스크립트 (답변 생성 호출 없음).

사용 예:
    python exp_max_per_doc.py --eval-file data/eval_retrieval_yjk_e2e24.json --max-per-doc 2 3
    RETRIEVAL_RERANK=lexical python exp_max_per_doc.py --eval-file ... --max-per-doc 2 3

기준선(문서당 상한 없음)은 항상 먼저 실행되고, --max-per-doc 값마다 같은 평가셋으로 비교합니다.
리랭킹은 기본으로 꺼져 있으며 환경 변수 RETRIEVAL_RERANK, RETRIEVAL_CANDIDATES로 켭니다 (retrieval.py 참고).
설정마다 질문을 다시 임베딩하므로 임베딩 API를 (설정 수 × 질문 수)번 호출하고, 지연에는 그 시간이 포함됩니다.
"""
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 평가 JSON의 키 이름 (data/eval_retrieval_yjk*.json 기준). 다른 평가 파일을 쓰면 맞게 수정하세요.
QUESTION_KEY = "question"
GOLD_KEY = "expected_doc_ids"        # 정답 문서 ID (여러 개면 리스트도 가능)
FILTER_KEY = "filters"     # 질문별 필터가 있는 경우


def evaluate_cases(cases, search, top_k):
    """search(question, filters) -> 점수순 hits 리스트(각 hit에 doc_id).

    recall·mrr은 정답 문서가 하나라도 나오면 성공으로 보므로 문서당 상한의 효과가 잘 드러나지 않습니다.
    그래서 top-k 구성을 보는 지표를 함께 계산합니다.
      recall_frac: 정답 문서 중 찾은 비율 (정답 문서가 여러 개인 질문에서 일부만 찾은 경우를 구분)
      unique_docs: top-k 안의 고유 문서 수 평균
      non_gold_share: top-k에서 정답 문서가 아닌 청크가 차지하는 비율 평균
      gold_chunks: top-k 안의 정답 문서 청크 수 평균
      gold_chunks_ge2: 정답 문서 청크가 2개 이상 들어온 질문 수
      short: top-k보다 적게 반환된 질문 수
    """
    found, rr_sum, times, misses = 0, 0.0, [], []
    frac_sum, unique_sum, non_gold_sum, gold_sum, gold_ge2, short = 0.0, 0, 0.0, 0, 0, 0
    for number, case in enumerate(cases):
        start = time.perf_counter()
        hits = search(case[QUESTION_KEY], case.get(FILTER_KEY) or {})
        times.append((time.perf_counter() - start) * 1000)
        ids = [hit["doc_id"] for hit in hits[:top_k]]
        gold = case[GOLD_KEY]
        gold = set(gold) if isinstance(gold, (list, tuple, set)) else {gold}
        if not gold:
            raise ValueError(f"{number}번 질문에 정답 문서 ID가 없습니다.")
        rank = next((i + 1 for i, doc_id in enumerate(ids) if doc_id in gold), None)
        if rank:
            found += 1
            rr_sum += 1 / rank
        else:
            misses.append(number)  # 놓친 질문 번호 (원인 분석용)
        gold_chunks = sum(doc_id in gold for doc_id in ids)
        frac_sum += len(gold & set(ids)) / len(gold)
        unique_sum += len(set(ids))
        non_gold_sum += (len(ids) - gold_chunks) / len(ids) if ids else 0.0
        gold_sum += gold_chunks
        gold_ge2 += gold_chunks >= 2
        short += len(ids) < top_k
    n = len(cases)
    return {"recall": found / n, "mrr": rr_sum / n, "median_ms": statistics.median(times),
            "cases": n, "misses": misses,
            "recall_frac": frac_sum / n, "unique_docs": unique_sum / n, "non_gold_share": non_gold_sum / n,
            "gold_chunks": gold_sum / n, "gold_chunks_ge2": gold_ge2, "short": short}


def main():
    parser = argparse.ArgumentParser(description="검색 단계 실험")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-per-doc", type=int, nargs="*", default=[],
                        help="비교할 문서당 최대 청크 수 (기준선은 항상 포함)")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes/parsing-v2-yjk")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    # API를 호출하기 전에 입력을 먼저 확인합니다. (기준선을 돌린 뒤에 실패하면 비용만 듭니다.)
    if args.top_k < 1 or any(limit < 1 for limit in args.max_per_doc):
        parser.error("top-k, max-per-doc은 1 이상이어야 합니다.")
    cases = json.loads(args.eval_file.read_text(encoding="utf-8"))
    if isinstance(cases, dict):  # {"cases": [...]} 또는 {"questions": [...]} 구조도 허용
        cases = cases.get("cases") or cases.get("questions")
    if not isinstance(cases, list) or not cases:
        parser.error("평가 파일에 질문 목록이 없습니다.")

    from dotenv import load_dotenv
    from openai import OpenAI
    from embedding import load_index
    from retrieval import retrieve

    load_dotenv(ROOT / ".env", override=False)  # 키는 이 컴퓨터의 .env에서만 읽습니다.
    if not os.getenv("OPENAI_API_KEY"):
        parser.error(".env 또는 환경 변수에 OPENAI_API_KEY를 설정하세요.")
    client = OpenAI(timeout=60.0, max_retries=2)
    index, chunks, config = load_index(args.index_dir)

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
        print(f"  구성: Recall(비율) {result['recall_frac']:.3f} / 고유 문서 {result['unique_docs']:.2f}개 / "
              f"정답 외 청크 {result['non_gold_share']:.1%} / 정답 청크 평균 {result['gold_chunks']:.2f}개 / "
              f"정답 청크 2개 이상 {result['gold_chunks_ge2']}건 / {args.top_k}개 미만 반환 {result['short']}건")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.results_dir / f"retrieval_exp_{stamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    # 인덱스와 리랭킹 설정에 따라 수치가 달라지므로 함께 남깁니다.
    settings = {**config, "index_dir": str(args.index_dir),
                **{key: os.getenv(key) for key in ("RETRIEVAL_RERANK", "RETRIEVAL_CANDIDATES", "RETRIEVAL_RERANK_MODEL")}}
    output.write_text(json.dumps({"eval_file": str(args.eval_file), "top_k": args.top_k, "settings": settings,
                                  "results": results}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"결과 저장: {output}")


if __name__ == "__main__":
    main()
