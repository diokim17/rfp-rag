"""evaluate 결과 JSON에서 검색 신호(문서 수·점수)와 답변 정답 여부의 관계를 봅니다 (조건 기반 라우팅 설계용).

신호는 a_retrieval.retrieval_signals로 sources에서 다시 계산하므로 signals가 없는 예전 결과에도 쓸 수 있습니다.
API·모델 호출 없음. 본문·질문은 출력하지 않고 문항 ID와 숫자만 출력합니다.

    python experiments/answer_signals_yjk.py results/a/evaluate_yjk-0040_....json
"""

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from statistics import mean
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from a_retrieval import retrieval_signals

SIGNALS = ("docs", "top_doc_share", "best_score", "rerank_gap")
# 구간 경계. 재정렬 점수 범위는 실제 결과를 보고 다시 맞춥니다.
EDGES = {"docs": [1, 3, 4, 5], "top_doc_share": [0.3, 0.5], "best_score": [0.6, 0.65, 0.7],
         "rerank_gap": [0.5, 1, 2, 4]}


def label(record):
    """correct(1점)·partial·wrong·refused(답할 수 있는데 거절)·unanswerable(답 없는 문항)·ungraded."""
    if not record["answerable"]:
        return "unanswerable"
    if record["refused"]:
        return "refused"
    score = record["answer_score"]
    if score is None:
        return "ungraded"
    return "correct" if score == 1 else "partial" if score > 0 else "wrong"


def context_coverage(record):
    """정답 키워드 중 검색 자료(청크 본문 + 모델에 함께 가는 사업 정보)에 있는 비율. 키워드가 없으면 None.

    답변 키워드 포함률(keyword_coverage)과 같은 방식(NFC·대소문자 무시)으로 셉니다.
    자료에 있는데 답이 틀렸으면 생성 쪽, 자료에 없으면 검색(청크 선택) 쪽 문제로 봅니다.
    """
    words = record["expected_keywords"]
    if not words:
        return None
    normalize = lambda text: unicodedata.normalize("NFC", text).casefold()
    context = normalize(" ".join(hit["text"] + " " + " ".join(map(str, hit["metadata"].values()))
                                 for hit in record["sources"]))
    return sum(normalize(word) in context for word in words) / len(words)


def excerpt_coverage(excerpt, record):
    """평가셋의 정답 원문 조각(reference_excerpt, LLM 판정 문항에만 있음)이 검색 청크 본문에 들어간 비율. 없으면 None.

    공백을 지운 뒤 조각을 20자 창(10자 간격)으로 나눠, 청크 본문에 그대로 있는 창의 비율을 셉니다.
    정답 부분이 검색됐으면 1에 가깝고, 청크 경계에 걸쳐 일부만 들어오면 그 사이 값이 나옵니다.
    """
    if not excerpt:
        return None
    squeeze = lambda text: re.sub(r"\s+", "", unicodedata.normalize("NFC", text))
    excerpt, context = squeeze(excerpt), squeeze(" ".join(hit["text"] for hit in record["sources"]))
    windows = [excerpt[i:i + 20] for i in range(0, max(len(excerpt) - 20, 0) + 1, 10)]
    return sum(window in context for window in windows) / len(windows)


def cause(row):
    """정답이 아닌 답할 수 있는 문항의 원인 추정: 문서 누락·청크 누락(검색) / 생성 / 미확인.

    키워드 문항은 키워드가 모두 자료에 있었는지(ctx_kw)로 가릅니다. LLM 판정 문항은 정답 원문 조각이 절반 이상
    자료에 있었으면(ctx_ref) 생성으로 보고, 그보다 낮으면 미확인으로 둡니다. yjk-0010(A)에서 ctx_ref가 0에 가까운데도
    정답인 문항이 12개 중 3개(yA004·yB044 0, yC059 0.018) 있어서, 낮은 값만으로 청크 누락이라고 할 수 없었습니다
    (답에 필요한 부분이 조각의 일부뿐이거나, 조각과 파싱된 청크의 글자가 다를 수 있음).
    """
    if row["label"] in {"correct", "unanswerable", "ungraded"}:
        return "-"
    if row["recall"] == 0:
        return "문서 누락"
    if row["ctx_kw"] is not None:
        return "청크 누락" if row["ctx_kw"] < 1 else "생성"
    if row["ctx_ref"] is not None and row["ctx_ref"] >= 0.5:
        return "생성"
    return "미확인"


def bucket(value, edges):
    if value is None:
        return "없음"
    for edge in edges:
        if value < edge:
            return f"<{edge}"
    return f">={edges[-1]}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json")
    parser.add_argument("--eval-file", default=ROOT / "data/eval_v4.json",
                        help="정답 원문 조각(reference_excerpt)을 읽을 평가셋. 결과 JSON을 만든 평가셋과 같아야 합니다.")
    args = parser.parse_args()
    with open(args.result_json, encoding="utf-8") as f:
        records = json.load(f)["records"]
    with open(args.eval_file, encoding="utf-8") as f:
        excerpts = {case["id"]: case.get("reference_excerpt") for case in json.load(f)}

    rows = [{"id": r["case_id"], "label": label(r), "score": r["answer_score"], "recall": r["recall_at_k"],
             "chunk_hit": r["answer_chunk_hit"], "ctx_kw": context_coverage(r), "ans_kw": r["keyword_coverage"],
             "ctx_ref": excerpt_coverage(excerpts[r["case_id"]], r), **retrieval_signals(r["sources"])}
            for r in records]
    for row in rows:
        row["cause"] = cause(row)

    columns = ("id", "label", "cause", "score", "recall", "chunk_hit", "ctx_kw", "ans_kw", "ctx_ref", *SIGNALS)
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    # 정답이 아닌 답할 수 있는 문항 → 정답 → 답 없는 문항 순서 (head로 잘라도 오답이 먼저 보이게)
    order = lambda row: (row["label"] == "unanswerable", row["label"] == "correct", str(row["id"]))
    for row in sorted(rows, key=order):
        print("| " + " | ".join(fmt(row[k]) for k in columns) + " |")

    # 신호 구간별 평균 점수: 답할 수 있는 문항만 (거절은 0점으로 셈)
    answerable = [row for row in rows if row["label"] not in {"unanswerable", "ungraded"}]
    print(f"\n답할 수 있는 문항 {len(answerable)}개 기준 구간별 평균 점수")
    for key in SIGNALS:
        groups = {}
        for row in answerable:
            groups.setdefault(bucket(row[key], EDGES[key]), []).append(row["score"] or 0)
        print(f"- {key}: " + ", ".join(f"{k}: {mean(v):.3f} ({len(v)}개)" for k, v in sorted(groups.items())))
    print("\n결과별 개수:", dict(Counter(row["label"] for row in rows)))
    print("정답이 아닌 문항 원인 추정:", dict(Counter(row["cause"] for row in rows if row["cause"] != "-")))


if __name__ == "__main__":
    main()
