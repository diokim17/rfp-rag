"""시나리오 A 호환 이름. 구현은 A·B 공통인 common_query_rewrite.py 하나로 합쳤습니다(같은 모듈 객체)."""
import sys

import common_query_rewrite

sys.modules[__name__] = common_query_rewrite
