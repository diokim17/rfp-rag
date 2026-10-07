"""김연주: rerank_grid_yjk.py 결과 JSON으로 PR 댓글(Markdown)을 자동 생성합니다.

  python experiments/rerank_grid_pr_comment_yjk.py                      # 가장 최근 results/rerank_grid_*.json
  python experiments/rerank_grid_pr_comment_yjk.py --result results/rerank_grid_20261006T025155Z.json

출력은 화면과 results/pr_comment_rerank_grid_<측정 시각>.md(Git 제외)에 저장합니다.
수치에서 바로 나오는 사실만 쓰며, 후보 수 권장 같은 판단은 직접 덧붙이세요.
질문·원문·필터 값은 넣지 않고 질문 ID와 유형만 사용합니다.
"""

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from parsing import read_json  # noqa: E402

BRANCH = "origin/retrieval/max-per-doc"


def fmt_ms(value):
    return f"{value / 1000:.2f}초" if value >= 1000 else f"{value:.0f}ms"


def span(values, fmt):
    low, high = min(values), max(values)
    return fmt(low) if fmt(low) == fmt(high) else f"{fmt(low)}~{fmt(high)}"


def new_commits():
    """아직 원격 브랜치에 없는, 이 브랜치에서 만든 커밋 (오래된 순). 병합으로 들어온 남의 커밋은 제외.
    git이 없거나 원격 브랜치를 모르면 빈 목록."""
    try:
        out = subprocess.check_output(["git", "log", "--first-parent", "--reverse", "--format=%h %s", f"{BRANCH}..HEAD"],
                                      cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.splitlines() if line.strip()]


def build_comment(result, cases):
    settings, summary = result["settings"], result["summary"]
    top_ks = sorted({row["top_k"] for row in summary})
    groups = defaultdict(dict)  # (리랭킹, 후보 수) → {top_k: row}
    for row in summary:
        groups[(row["rerank"], row["candidates"])][row["top_k"]] = row

    lines = []
    commits = new_commits()
    if commits:
        lines += ["## 추가 커밋", ""] + [f"- `{line.split(' ', 1)[0]}` {line.split(' ', 1)[1]}" for line in commits] + [""]

    k_label = " / ".join(f"@{k}" for k in top_ks)
    lines += [f"## 리랭킹 × 후보 수 × top-k 비교: {settings['eval_file']} ({settings['cases']}문항)", "",
              f"인덱스: 청크 {settings['chunk_count']:,}개 ({settings['chunk_size']}/{settings['chunk_overlap']}, "
              f"{settings.get('chunking_strategy', 'fixed')}), `{settings['embedding_model']}`, "
              f"측정 {settings['measured_at_utc']} (UTC)", "",
              f"| 리랭킹 | 후보 수 | Recall{k_label} | MRR{k_label} | 질문당 평균 시간 |",
              "| --- | ---: | --- | --- | ---: |"]
    for (mode, candidates), by_k in groups.items():
        rows = [by_k[k] for k in top_ks]
        lines.append(f"| {'없음' if mode == 'none' else mode} | {candidates or '-'} | "
                     + " / ".join(f"{row['recall']:.3f}" for row in rows) + " | "
                     + " / ".join(f"{row['mrr']:.3f}" for row in rows) + " | "
                     + fmt_ms(sum(row["mean_ms"] for row in rows) / len(rows)) + " |")

    # 사실만 자동 요약
    lines += [""]
    base = [row for row in summary if row["rerank"] == "none"]
    reranked = [row for row in summary if row["rerank"] != "none"]
    if base and reranked:
        lines.append(f"- 리랭킹 없는 검색의 MRR은 {span([r['mrr'] for r in base], lambda v: f'{v:.3f}')}, "
                     f"리랭킹을 쓴 설정의 MRR은 {span([r['mrr'] for r in reranked], lambda v: f'{v:.3f}')}입니다.")
    missed = defaultdict(int)
    for row in summary:
        for case_id in row["misses"]:
            missed[case_id] += 1
    if missed:
        details = []
        for case_id, count in sorted(missed.items(), key=lambda item: -item[1]):
            case = cases.get(case_id, {})
            extra = [case["type"]] if case.get("type") else []
            if case.get("expected_doc_ids"):
                extra.append(f"정답 문서 {len(case['expected_doc_ids'])}개")
            if case.get("filters"):
                extra.append("필터 있음")
            details.append(f"{case_id}({', '.join(extra)}) {count}/{len(summary)}개 설정" if extra
                           else f"{case_id} {count}/{len(summary)}개 설정")
        lines.append("- 정답 문서를 다 찾지 못한 질문: " + ", ".join(details) + ".")
    else:
        lines.append("- 모든 설정에서 모든 질문의 정답 문서를 찾았습니다.")
    for mode in dict.fromkeys(row["rerank"] for row in reranked):
        rows = [row for row in reranked if row["rerank"] == mode]
        by_candidates = defaultdict(list)
        for row in rows:
            by_candidates[row["candidates"]].append(row["mean_ms"])
        times = ", ".join(f"후보 {c} {fmt_ms(sum(v) / len(v))}" for c, v in sorted(by_candidates.items()))
        lines.append(f"- {mode} 질문당 평균 시간: {times}.")
    if settings.get("cross_encoder_load_ms"):
        lines.append(f"- cross-encoder 모델 첫 로드 {fmt_ms(settings['cross_encoder_load_ms'])}는 측정에서 제외했습니다.")
    lines.append("- 처리 시간은 질문 임베딩 캐시를 쓴 검색·리랭킹 시간이며 임베딩 API 지연은 포함하지 않습니다.")
    lines += [f"- 전체 표: `results/reports/rerank_grid_yjk_{Path(settings['eval_file']).stem}.md`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", type=Path, help="rerank_grid_*.json (생략: 가장 최근 파일)")
    parser.add_argument("--eval-file", type=Path, help="질문 유형을 읽을 평가 파일 (생략: data/<결과의 eval_file>)")
    parser.add_argument("--output", type=Path, help="저장할 Markdown 경로 (생략: results/pr_comment_rerank_grid_<시각>.md)")
    args = parser.parse_args()

    path = args.result or max((ROOT / "results").glob("rerank_grid_*.json"), default=None)
    if path is None or not path.is_file():
        parser.error("결과 파일이 없습니다. 먼저 experiments/rerank_grid_yjk.py를 실행하세요.")
    result = read_json(path)
    eval_file = args.eval_file or ROOT / "data" / result["settings"]["eval_file"]
    cases = {case.get("id"): case for case in read_json(eval_file)} if eval_file.is_file() else {}

    comment = build_comment(result, cases)
    output = args.output or ROOT / f"results/pr_comment_rerank_grid_{result['settings']['measured_at_utc']}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(comment, encoding="utf-8")
    print(comment)
    print(f"저장: {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
