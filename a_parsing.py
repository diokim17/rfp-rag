"""시나리오 A 호환 이름. 구현은 A·B 공통인 common_parsing.py 하나로 합쳤습니다(같은 모듈 객체)."""
import sys

import common_parsing

sys.modules[__name__] = common_parsing
