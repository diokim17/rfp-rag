"""시나리오를 선택하는 호환 CLI. 구현은 a_run / b_run에서 독립적으로 관리합니다."""
import os
from pathlib import Path


def main():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
    scenario = os.getenv("RFP_SCENARIO", "B").strip().upper()
    if scenario == "A":
        from a_run import main as selected_main
    elif scenario == "B":
        from b_run import main as selected_main
    else:
        raise SystemExit("RFP_SCENARIO는 A 또는 B여야 합니다.")
    selected_main()


if __name__ == "__main__":
    main()
