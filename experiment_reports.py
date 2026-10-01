"""본문이 제외된 팀 공유용 평가 요약."""

import hashlib
from pathlib import Path


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def save_report(directory, timestamp, summary, settings, experiment, usage):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # 자동 생성 ID는 영문 이니셜과 숫자뿐이므로 파일명에도 안전하게 사용.
    experiment_id = experiment.get("experiment_id", "summary")
    path = directory / f"evaluate_{experiment_id}_{timestamp}.md"
    # 사용자 입력이 Markdown의 별도 줄이나 HTML이 되지 않게 인라인 코드 처리.
    def literal(value):
        return "`" + str(value).replace("|", "&#124;").replace("`", "'").replace("\n", " ").replace("\r", " ") + "`"
    lines = ["# 평가 실험 요약", "", f"실행 시각(UTC): {timestamp}", "",
             "## 실험 정보", "", "| 항목 | 값 |", "| --- | --- |"]
    for key, value in experiment.items():
        lines.append(f"| {literal(key)} | {literal(value)} |")
    lines += ["", "## 설정", "", "| 항목 | 값 |", "| --- | --- |"]
    # 필터 값과 경로/본문은 공유하지 않음.
    for key in ("embedding_model", "generation_model", "dimension", "chunk_count", "chunk_size", "chunk_overlap", "top_k"):
        lines.append(f"| {key} | {literal(settings.get(key))} |")
    lines += ["", "## 지표", "", "| 지표 | 값 |", "| --- | --- |"]
    for key, value in summary.items():
        lines.append(f"| {literal(key)} | {literal(value)} |")
    lines += ["", "## 모델별 토큰 사용량", "", "| 모델 | 입력 | 출력 | 전체 |", "| --- | --- | --- | --- |"]
    for model, values in usage.items():
        lines.append(f"| {literal(model)} | {values.get('input', 0)} | {values.get('output', 0)} | {values.get('total', 0)} |")
    lines += ["", "키워드 포함률은 사실성 지표가 아닙니다. 토큰은 성공한 API 응답에서 받은 사용량이며 재시도·실패 과금은 포함되지 않을 수 있습니다.",
              "비용은 이 보고서에서 계산하지 않습니다. Langfuse의 모델 가격 기반 기록과 제공업체 사용량을 확인하세요.",
              "질문·원문·답변·필터 값·API 키는 이 보고서에 포함하지 않습니다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
