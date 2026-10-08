"""필터 값 표기 차이 보정(_resolve_filters)을 외부 API 없이 검증합니다.

정확히 일치하는 값이 있으면 예전과 같고, 없을 때만 정규화 포함 관계로 하나뿐인 값을 찾습니다.
"""

import tempfile
import unittest

from embedding import build_index, load_index
from retrieval import _resolve_filters, retrieve
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정
from test_retrieval_rerank import FakeClient, document

QUESTION = "사업 요구사항"


class FilterResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        # 발주 기관은 실제 CSV 표기를 따릅니다(괄호·도 이름·'서울특별시').
        build_index([
            document("rail", "예약발매시스템 개량", "한국철도공사 (용역)", "가 예약발매 요구사항"),
            document("bus", "버스정보시스템 구축", "경기도 평택시", "나 버스 요구사항"),
            document("city", "도시공사 홈페이지", "평택시도시공사", "다 홈페이지 요구사항"),
            document("seoul", "삭제지원시스템 통합", "서울특별시 여성가족재단", "가 삭제지원 요구사항"),
            document("water", "건설통합시스템 고도화", "한국수자원공사", "나 건설 요구사항"),
            document("water2", "수문자료 재구축", "한국수자원공사 (용역)", "다 수문 요구사항"),
        ], cls.client, cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def docs(self, filters):
        hits = retrieve(QUESTION, self.client, self.index, self.chunks, self.config, 10, filters)
        return sorted({hit["doc_id"] for hit in hits})

    def test_exact_value_is_kept_even_if_others_contain_it(self):
        self.assertEqual(self.docs({"발주 기관": "한국수자원공사"}), ["water"])
        self.assertEqual(self.docs({"발주 기관": "한국수자원공사 (용역)"}), ["water2"])

    def test_spacing_and_brackets_are_ignored(self):
        self.assertEqual(self.docs({"발주 기관": "한국철도공사"}), ["rail"])
        self.assertEqual(_resolve_filters({"발주 기관": "한국철도공사"}, self.chunks),
                         {"발주 기관": "한국철도공사 (용역)"})

    def test_seoul_abbreviation(self):
        self.assertEqual(self.docs({"발주 기관": "서울시여성가족재단"}), ["seoul"])
        self.assertEqual(self.docs({"발주 기관": "서울특별시여성가족재단"}), ["seoul"])

    def test_ambiguous_value_returns_nothing(self):
        # '평택시'는 '경기도 평택시'와 '평택시도시공사' 둘 다에 들어 있어 고르지 않습니다.
        self.assertEqual(self.docs({"발주 기관": "평택시"}), [])
        self.assertEqual(self.docs({"발주 기관": "경기도평택시"}), ["bus"])

    def test_unmatched_or_empty_value_returns_nothing_without_api_call(self):
        calls = self.client.calls
        for value in ("없는기관", "()", " "):
            self.assertEqual(self.docs({"발주 기관": value}), [], value)
        self.assertEqual(self.docs({"없는 키": "한국철도공사"}), [])
        self.assertEqual(self.client.calls, calls)

    def test_every_filter_must_still_match(self):
        self.assertEqual(self.docs({"발주 기관": "한국철도공사", "사업명": "예약발매시스템 개량"}), ["rail"])
        self.assertEqual(self.docs({"발주 기관": "한국철도공사", "사업명": "버스정보시스템 구축"}), [])

    def test_input_filters_are_not_modified(self):
        filters = {"발주 기관": "한국철도공사"}
        self.docs(filters)
        self.assertEqual(filters, {"발주 기관": "한국철도공사"})


if __name__ == "__main__":
    unittest.main()
