"""검색 방식과 무관한 단계 간 계약 테스트용: retrieve 기본값을 예전 baseline(코사인 검색만)으로 고정합니다.

기본값(하이브리드 + cross-encoder + 문서당 상한 2 + 범위 밖 차단)은 실제 모델이 필요해 오프라인 테스트에서 쓸 수 없고,
torch 설치 여부에 따라 결과가 달라지지 않도록 모듈 상수를 바꿉니다(환경 변수를 비우는 테스트에도 유지됩니다).
테스트 모듈에서 `from retrieval_baseline import setUpModule  # noqa: F401`로 가져다 씁니다.
"""

import unittest
from unittest.mock import patch

import a_retrieval
import b_retrieval

BASELINE = {"DEFAULT_RERANK": "none", "DEFAULT_HYBRID": False, "DEFAULT_MAX_PER_DOC": None, "DEFAULT_BM25_PREFIX": False,
            "DEFAULT_EXPAND": 0, "DEFAULT_MIN_SCORE": None}


def setUpModule():
    # A는 범위 밖 차단 기준(DEFAULT_MIN_SCORE_A, 0.53)과 재정렬 절 제목(DEFAULT_RERANK_SECTION)도 끕니다.
    for module, extra in ((a_retrieval, {"DEFAULT_MIN_SCORE_A": None, "DEFAULT_RERANK_SECTION": False}),
                          (b_retrieval, {})):
        # 사업 하나 질문의 상한 해제(B 기본 켬)도 baseline에서는 끕니다.
        if hasattr(module, "DEFAULT_UNCAP_NAMED"):
            extra = {**extra, "DEFAULT_UNCAP_NAMED": False}
        patcher = patch.multiple(module, **BASELINE, **extra)
        patcher.start()
        unittest.addModuleCleanup(patcher.stop)
