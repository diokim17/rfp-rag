"""시나리오 A 호환 이름. 구현은 A·B 공통인 common_evaluation.py 하나로 합쳤습니다(같은 모듈 객체)."""
import sys

import common_evaluation

sys.modules[__name__] = common_evaluation
