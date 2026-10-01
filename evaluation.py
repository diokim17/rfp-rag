"""김시현: 수동 정답 데이터로 검색 Recall@k와 답변 키워드 포함률 평가."""

import time
import unicodedata
from observability import observed


@observed("evaluate")
def evaluate(cases, answer_fn):
    """cases: [{question, expected_doc_ids, expected_keywords?, filters?}]."""
    if not cases:
        raise ValueError("평가 데이터가 비어 있습니다.")
    for case in cases:
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("평가 질문이 필요합니다.")
        ids = case.get("expected_doc_ids")
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
            raise ValueError("expected_doc_ids에 정답 문서 ID 목록이 필요합니다.")
        words = case.get("expected_keywords", [])
        if not isinstance(words, list) or not all(isinstance(x, str) and x.strip() for x in words):
            raise ValueError("expected_keywords는 비어 있지 않은 문자열 목록이어야 합니다.")
    records = []
    for case in cases:
        start = time.perf_counter()
        result = answer_fn(case["question"], case.get("filters"))
        expected = set(case["expected_doc_ids"])
        retrieved = {source["doc_id"] for source in result["sources"]}
        words = case.get("expected_keywords", [])
        normalize = lambda text: unicodedata.normalize("NFC", text).casefold()
        records.append({
            "recall_at_k": len(expected & retrieved) / len(expected),
            "keyword_coverage": sum(normalize(word) in normalize(result["answer"]) for word in words) / len(words) if words else None,
            "latency_seconds": time.perf_counter() - start,
            "expected_doc_ids": sorted(expected), "expected_keywords": words, **result,
        })
    keyword_scores = [r["keyword_coverage"] for r in records if r["keyword_coverage"] is not None]
    return {"summary": {
        "count": len(records),
        "recall_at_k": sum(r["recall_at_k"] for r in records) / len(records),
        "keyword_coverage": sum(keyword_scores) / len(keyword_scores) if keyword_scores else None,
        "latency_seconds": sum(r["latency_seconds"] for r in records) / len(records),
    }, "records": records}
