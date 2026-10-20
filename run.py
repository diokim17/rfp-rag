"""시나리오 선택 실행기: python run.py [명령 ...] 을 실행하면 A·B·A+B 중 하나를 고릅니다.

A(로컬 모델, transformers 5.1)와 B(OpenAI API, transformers 4.57)는 패키지 버전이 달라 한 가상환경에서
함께 돌 수 없으므로, 고른 시나리오의 가상환경 파이썬으로 a_run.py·b_run.py를 별도 프로세스로 실행합니다.
터미널에서는 항상 묻고, RFP_SCENARIO(A·B·AB, 없으면 B)는 Enter만 눌렀을 때의 기본값입니다.
입력을 받을 수 없는 환경(배치 실행 등)에서는 묻지 않고 그 기본값으로 실행합니다.
가상환경은 .venv-a(A), rag-b 또는 .venv(B) 순서로 찾고, RFP_PYTHON_A·RFP_PYTHON_B로 바꿀 수 있습니다.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CHOICES = {"1": "A", "2": "B", "3": "AB", "A": "A", "B": "B", "AB": "AB", "A+B": "AB"}
VENVS = {"A": (".venv-a",), "B": ("rag-b", ".venv")}
SUMMARY_ROWS = (("답변 정확도", "answer_accuracy"), ("내용 정확도", "answer_content_accuracy"),
                ("거절 정확도", "refusal_accuracy"), ("오거절률", "false_refusal_rate"),
                ("Recall@k", "recall_at_k"), ("MRR@k", "mrr_at_k"),
                ("응답 p50(초)", "latency_p50_seconds"), ("응답 p95(초)", "latency_p95_seconds"))


def choose_scenario():
    value = default = os.getenv("RFP_SCENARIO", "").strip().upper() or "B"
    if sys.stdin.isatty():
        print("실행할 시나리오를 선택하세요\n  1) A    서버에서 모델 직접 실행\n  2) B    외부 API(OpenAI)\n  3) A+B  A와 B를 차례로 실행")
        value = input(f"> (Enter: {default}) ").strip().upper() or default
    if value not in CHOICES:
        raise SystemExit("A, B, AB(또는 1, 2, 3) 중에서 고르세요.")
    return CHOICES[value]


def python_for(scenario):
    if os.getenv(f"RFP_PYTHON_{scenario}"):
        return os.environ[f"RFP_PYTHON_{scenario}"]
    for venv in VENVS[scenario]:
        for name in ("bin/python", "Scripts/python.exe"):
            if (ROOT / venv / name).is_file():
                return str(ROOT / venv / name)
    return sys.executable


def command_for(scenario, args):
    return [python_for(scenario), str(ROOT / f"{scenario.lower()}_run.py"), *args]


def env_for(scenario):
    return {**os.environ, "RFP_SCENARIO": scenario}


def run_and_capture(scenario, args):
    """출력을 그대로 보여 주면서 '결과 저장: ….json' 경로를 찾아 돌려줍니다."""
    print(f"\n===== 시나리오 {scenario} =====", flush=True)
    process = subprocess.Popen(command_for(scenario, args), cwd=ROOT, env=env_for(scenario),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    saved = None
    for line in process.stdout:
        print(line, end="", flush=True)
        match = re.match(r"결과 저장: (.+\.json)$", line.strip())
        saved = match.group(1) if match else saved
    if process.wait():
        raise SystemExit(f"시나리오 {scenario} 실행 실패 (종료 코드 {process.returncode})")
    return saved


def print_summary(saved):
    summaries = {}
    for scenario, path in saved.items():
        result = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        if "summary" in result:
            summaries[scenario] = result["summary"]
    if len(summaries) < 2:
        return
    print("\n| 지표 | A | B |\n| --- | ---: | ---: |")
    for label, key in SUMMARY_ROWS:
        cells = [f"{summaries[s][key]:.3f}" if isinstance(summaries[s].get(key), (int, float)) else "-" for s in ("A", "B")]
        print(f"| {label} | {cells[0]} | {cells[1]} |")
    print(f"\nA 결과: {saved['A']}\nB 결과: {saved['B']}")


def main():
    try:  # 시나리오만 고르므로 가상환경 밖 python으로 실행해도 됩니다.
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    scenario = choose_scenario()
    args = sys.argv[1:]
    if scenario != "AB":
        raise SystemExit(subprocess.call(command_for(scenario, args), cwd=ROOT, env=env_for(scenario)))
    print_summary({s: run_and_capture(s, args) for s in ("A", "B")})


if __name__ == "__main__":
    main()
