"""시나리오 B 호환 이름. 구현은 A·B 공통인 common_experiment_reports.py 하나로 합쳤습니다(같은 모듈 객체)."""
import sys

import common_experiment_reports

sys.modules[__name__] = common_experiment_reports
