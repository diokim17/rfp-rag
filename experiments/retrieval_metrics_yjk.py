"""김연주: 검색 평가 지표 Recall@k·MRR@k (문서 단위 정답 기준).

정의는 평가 담당 evaluation.py의 recall_at_k와 같고, 검색 실험 스크립트들이 이 파일을 공통으로 씁니다.

  Recall@k = 상위 k개 결과에 나온 정답 문서 수 / 정답 문서 수
  MRR@k    = 상위 k개 결과에서 처음 나온 정답 문서 순위의 역수 (없으면 0). 순위는 청크 순위입니다.

저장된 rerank_grid 결과로 지표를 다시 계산해 저장할 수도 있습니다 (API·모델 호출 없음):

  python experiments/retrieval_metrics_yjk.py                       # 가장 최근 results/rerank_grid_*.json
  python experiments/retrieval_metrics_yjk.py --result results/rerank_grid_20261006T043614Z.json
"""

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parsing import read_json, write_json  # noqa: E402


def _check(gold, k):
    gold = set(gold)
    if not gold:
        raise ValueError("정답 문서 ID가 비어 있습니다.")
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")
    return gold


def recall_at_k(gold, retrieved, k):
    """gold: 정답 문서 ID들, retrieved: 점수순 문서 ID 목록(청크마다 하나, 중복 가능)."""
    gold = _check(gold, k)
    return len(gold & set(retrieved[:k])) / len(gold)


def mrr_at_k(gold, retrieved, k):
    gold = _check(gold, k)
    return next((1 / rank for rank, doc in enumerate(retrieved[:k], 1) if doc in gold), 0.0)


def score_case(gold, retrieved, k):
    """질문 하나의 {recall, mrr}."""
    return {"recall": recall_at_k(gold, retrieved, k), "mrr": mrr_at_k(gold, retrieved, k)}


def mean_scores(rows):
    """질문별 {recall, mrr} 목록의 평균과 정답 문서를 다 찾지 못한 질문 ID."""
    if not rows:
        raise ValueError("평가할 질문이 없습니다.")
    return {"recall": statistics.fmean(row["recall"] for row in rows),
            "mrr": statistics.fmean(row["mrr"] for row in rows),
            "cases": len(rows),
            "misses": [row.get("id") for row in rows if row["recall"] < 1]}


def rescore(result, cases):
    """rerank_grid 결과의 질문별 hits로 지표를 다시 계산합니다. 반환: 설정별 요약 목록."""
    gold = {case.get("id"): case["expected_doc_ids"] for case in cases}
    timing = {(row["rerank"], row["candidates"], row["top_k"]): row for row in result["summary"]}
    groups = defaultdict(list)
    for record in result["records"]:
        if record["id"] not in gold:
            raise ValueError(f"평가 파일에 없는 질문 ID: {record['id']}")
        key = (record["rerank"], record["candidates"], record["top_k"])
        groups[key].append({"id": record["id"], **score_case(gold[record["id"]], record["hits"], record["top_k"])})
    summary = []
    for key, rows in groups.items():
        row = {"rerank": key[0], "candidates": key[1], "top_k": key[2], **mean_scores(rows)}
        saved = timing.get(key, {})
        row["mean_ms"], row["median_ms"] = saved.get("mean_ms"), saved.get("median_ms")
        row["matches_saved"] = (abs(row["recall"] - saved.get("recall", -1)) < 1e-12
                                and abs(row["mrr"] - saved.get("mrr", -1)) < 1e-12)
        summary.append(row)
    return summary


def report_lines(title, meta, summary):
    lines = [f"# 검색 평가 지표 (Recall@k, MRR@k): {title}", "", "| 설정 | 값 |", "| --- | --- |"]
    lines += [f"| {key} | `{value}` |" for key, value in meta.items()]
    lines += ["", "| 리랭킹 | 후보 수 | top-k | Recall@k | MRR@k | 질문당 평균(ms) | 정답 누락 질문 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for row in summary:
        ms = "-" if row["mean_ms"] is None else f"{row['mean_ms']:.1f}"
        lines.append(f"| {row['rerank']} | {row['candidates'] or '-'} | {row['top_k']} | {row['recall']:.3f} | "
                     f"{row['mrr']:.3f} | {ms} | {', '.join(map(str, row['misses'])) or '-'} |")
    lines += ["", "Recall@k = 상위 k개에 나온 정답 문서 수 / 정답 문서 수. "
              "MRR@k = 상위 k개에서 처음 나온 정답 문서 순위의 역수(없으면 0). 문서 단위 정답 기준입니다.",
              "처리 시간은 원래 실험에서 잰 값(질문 임베딩 캐시 사용, 모델 로드 제외)입니다.", ""]
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, help="rerank_grid_*.json (생략: 가장 최근 파일)")
    parser.add_argument("--eval-file", type=Path, help="정답이 있는 평가 파일 (생략: data/<결과의 eval_file>)")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "results/reports")
    args = parser.parse_args()

    path = args.result or max((ROOT / "results").glob("rerank_grid_*.json"), default=None)
    if path is None or not path.is_file():
        parser.error("결과 파일이 없습니다. 먼저 experiments/rerank_grid_yjk.py를 실행하세요.")
    result = read_json(path)
    eval_file = args.eval_file or ROOT / "data" / result["settings"]["eval_file"]
    if not eval_file.is_file():
        parser.error(f"평가 파일이 없습니다: {eval_file}")
    try:
        summary = rescore(result, read_json(eval_file))
    except ValueError as exc:
        parser.error(str(exc))

    settings = result["settings"]
    meta = {"source": path.name, "eval_file": eval_file.name, "cases": summary[0]["cases"],
            "evaluation_sha256": settings.get("evaluation_sha256"), "index_dir": settings.get("index_dir"),
            "chunk_count": settings.get("chunk_count"), "embedding_model": settings.get("embedding_model"),
            "measured_at_utc": settings.get("measured_at_utc")}
    output = args.results_dir / f"retrieval_metrics_{settings.get('measured_at_utc', 'result')}.json"
    write_json(output, {"settings": meta, "summary": summary})
    report = args.reports_dir / f"retrieval_metrics_yjk_{eval_file.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = report_lines(eval_file.stem, meta, summary)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[len(meta) + 5:-3]))
    mismatched = [row for row in summary if not row["matches_saved"]]
    print(f"\n원래 실험 값과 일치: {len(summary) - len(mismatched)}/{len(summary)}개 설정")
    print(f"결과 저장: {output}\n공유용 요약: {report}")


if __name__ == "__main__":
    main()
