"""김시현: 수동 정답 데이터로 검색 Recall@k와 답변 키워드 포함률 평가.

나상훈: 답이 없는 문항(answerable: false)은 Recall·키워드 대신 거절 여부로 채점하고(refusal_accuracy),
금액·날짜는 표기를 맞춘 뒤 비교합니다('1억 원'·'100,000,000원' → 100000000, '2025년 5월 31일' → 2025-05-31).
"""

import re
import time
import unicodedata
from observability import observed

# 거절로 보는 답변 문구: generation.py의 NO_HITS_ANSWER·범위 밖 거절 문구와 근거 부족 규칙("확인되지 않")
# ponytail: 문구 포함만 봅니다. 거절 문구와 지어낸 답을 함께 쓴 답변도 맞음으로 셉니다.
REFUSAL = re.compile(r"답변할 수 없|질문에만 답할 수|확인되지 않|확인할 수 없|찾을 수 없")
DATE = re.compile(r"(\d{4})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})\s*일?")
# 쉼표를 뺀 뒤 적용. '1억', '8700만', '2억 4천만', '2억 4290만'
# ponytail: '4천5백만'처럼 백·십 단위가 섞인 표기는 바꾸지 않습니다(평가셋 금액 키워드에 없음).
WON = re.compile(r"(?:(\d+(?:\.\d+)?)\s*억)?\s*(?:(\d+)\s*(천)?\s*만)?(\s*)")


def _won(match):
    eok, man, thousand, space = match.groups()
    if not eok and not man:
        return match[0]
    return str(round(float(eok or 0) * 10**8) + int(man or 0) * (10**7 if thousand else 10**4)) + space


def _normalize(text):
    text = unicodedata.normalize("NFC", text).casefold()
    text = DATE.sub(lambda m: f"{m[1]}-{int(m[2]):02d}-{int(m[3]):02d}", text)
    text = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    return WON.sub(_won, text)


@observed("evaluate")
def evaluate(cases, answer_fn):
    """cases: [{question, expected_doc_ids, expected_keywords?, filters?, answerable?}].
    answerable이 false인 문항은 expected_doc_ids가 비어도 되며, Recall·키워드 평균에서 빠지고 거절 정확도로만 셉니다."""
    if not cases:
        raise ValueError("평가 데이터가 비어 있습니다.")
    for case in cases:
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError("평가 질문이 필요합니다.")
        ids = case.get("expected_doc_ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) and x for x in ids) \
                or (not ids and case.get("answerable", True) is not False):
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
        answerable = case.get("answerable", True) is not False
        records.append({
            "recall_at_k": len(expected & retrieved) / len(expected) if answerable else None,
            "keyword_coverage": sum(_normalize(word) in _normalize(result["answer"]) for word in words) / len(words)
            if words and answerable else None,
            "refused": None if answerable else bool(REFUSAL.search(result["answer"])),
            "latency_seconds": time.perf_counter() - start,
            "expected_doc_ids": sorted(expected), "expected_keywords": words, **result,
        })
    mean = lambda values: sum(values) / len(values) if values else None
    recalls = [r["recall_at_k"] for r in records if r["recall_at_k"] is not None]
    summary = {
        "count": len(records),
        "recall_at_k": mean(recalls),
        "keyword_coverage": mean([r["keyword_coverage"] for r in records if r["keyword_coverage"] is not None]),
        "latency_seconds": mean([r["latency_seconds"] for r in records]),
    }
    refusals = [r["refused"] for r in records if r["refused"] is not None]
    if refusals:  # 답이 없는 문항이 있을 때만 추가해 기존 평가셋 결과 형식은 그대로 둡니다.
        summary.update(answerable_count=len(recalls), no_answer_count=len(refusals), refusal_accuracy=mean(refusals))
    return {"summary": summary, "records": records}
