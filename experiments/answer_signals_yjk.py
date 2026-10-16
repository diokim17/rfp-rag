"""evaluate 결과 JSON에서 검색 신호(문서 수·점수)와 답변 정답 여부의 관계를 봅니다 (조건 기반 라우팅 설계용).

신호는 a_retrieval.retrieval_signals로 sources에서 다시 계산하므로 signals가 없는 예전 결과에도 쓸 수 있습니다.
API·모델 호출 없음. 본문·질문은 출력하지 않고 문항 ID와 숫자만 출력합니다.

    python experiments/answer_signals_yjk.py results/a/evaluate_yjk-0040_....json
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

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
    args = parser.parse_args()
    with open(args.result_json, encoding="utf-8") as f:
        records = json.load(f)["records"]

    rows = [{"id": r["case_id"], "label": label(r), "score": r["answer_score"], "recall": r["recall_at_k"],
             "chunk_hit": r["answer_chunk_hit"], **retrieval_signals(r["sources"])} for r in records]

    columns = ("id", "label", "score", "recall", "chunk_hit", *SIGNALS)
    print("| " + " | ".join(columns) + " |")
    print("|" + "---|" * len(columns))
    fmt = lambda v: "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)
    for row in sorted(rows, key=lambda row: (row["label"] == "correct", str(row["id"]))):
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


if __name__ == "__main__":
    main()
