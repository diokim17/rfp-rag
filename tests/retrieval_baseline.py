"""검색 방식과 무관한 단계 간 계약 테스트용: retrieve 기본값을 예전 baseline(코사인 검색만)으로 고정합니다.

기본값(하이브리드 + cross-encoder + 문서당 상한 2)은 실제 모델이 필요해 오프라인 테스트에서 쓸 수 없고,
torch 설치 여부에 따라 결과가 달라지지 않도록 모듈 상수를 바꿉니다(환경 변수를 비우는 테스트에도 유지됩니다).
테스트 모듈에서 `from retrieval_baseline import setUpModule  # noqa: F401`로 가져다 씁니다.
"""

import unittest
from unittest.mock import patch

import retrieval

BASELINE = {"DEFAULT_RERANK": "none", "DEFAULT_HYBRID": False, "DEFAULT_MAX_PER_DOC": None, "DEFAULT_BM25_PREFIX": False,
            "DEFAULT_EXPAND": 0}


def setUpModule():
    patcher = patch.multiple(retrieval, **BASELINE)
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
