"""검색 기본값(하이브리드 + cross-encoder + 문서당 상한 2)을 외부 API·모델 다운로드 없이 검증합니다.

cross-encoder 점수는 가짜 함수로 바꾸고, 설치 여부는 _cross_encoder_available을 바꿔 흉내 냅니다.
"""

import os
import tempfile
import unittest
import warnings
from unittest.mock import patch

from embedding import build_index, load_index
import retrieval
from retrieval import retrieval_options, retrieve
from test_retrieval_rerank import FakeClient, document

QUESTION = "학사정보시스템 고도화 사업의 요구사항"
DEFAULTS = {"rerank": "cross-encoder", "candidates": 50, "max_per_doc": 2, "hybrid": True,
            "hybrid_vector_k": 100, "hybrid_bm25_k": 100, "rrf_k": 60, "rewrite": "off", "rewrite_model": None,
            "bm25_prefix": True, "within_doc": "off", "expand": 5}
BASELINE = {"rerank": "none", "hybrid": False, "max_per_doc": "none"}


def fake_cross_encoder(question, passages, model_name):
    """'학사'가 든 후보를 위로, 그중 본문이 짧을수록 위로 올리는 결정적인 점수."""
    return [("학사" in passage) * 10 - len(passage) / 1000 for passage in passages]


class DefaultRetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        # haksa 문서는 청크가 3개라 상한 2가 실제로 하나를 잘라야 합니다.
        build_index([
            document("haksa", "학사정보시스템 고도화", "나", "가 학사 일정 요구사항"),
            document("haksa", "학사정보시스템 고도화", "나", "가 학사 성적 요구사항 정의"),
            document("haksa", "학사정보시스템 고도화", "나", "가 학사 수강 신청 요구사항 정의서"),
            document("bus", "버스정보시스템 구축", "가", "나 버스 노선 안내와 제안 안내"),
            document("water", "용수공급 타당성조사", "다", "다 용수 공급 계획과 제안 안내"),
        ], cls.client, cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("RETRIEVAL_")},
                         clear=True)
        env.start()
        self.addCleanup(env.stop)
        available = patch.object(retrieval, "_cross_encoder_available", return_value=True)
        available.start()
        self.addCleanup(available.stop)

    def search(self, top_k=5, filters=None, **options):
        return retrieve(QUESTION, self.client, self.index, self.chunks, self.config, top_k, filters, **options)

    def test_default_options_are_hybrid_cross_encoder_cap2(self):
        self.assertEqual(retrieval_options(), DEFAULTS)

    def test_default_retrieve_runs_bm25_cross_encoder_and_cap(self):
        bm25 = patch.object(retrieval, "_bm25_ranking", wraps=retrieval._bm25_ranking)
        scorer = patch.object(retrieval, "_cross_encoder_scores", side_effect=fake_cross_encoder)
        with bm25 as bm25_spy, scorer as scorer_spy:
            hits = self.search(5)
        # 하이브리드 BM25(후보 100, 접두 색인)는 한 번, 나머지는 후보 확장의 문서 안 BM25(상위 문서 2개)
        hybrid_calls = [call for call in bm25_spy.call_args_list if call.args[3] == 100]
        self.assertEqual(len(hybrid_calls), 1)
        self.assertIs(hybrid_calls[0].args[4], True)
        self.assertEqual(len(bm25_spy.call_args_list) - 1, retrieval.EXPAND_DOCS)
        scorer_spy.assert_called_once()
        self.assertEqual([hit["doc_id"] for hit in hits], ["haksa", "haksa", "bus", "water"])  # 셋째 학사 청크는 상한으로 제외
        self.assertEqual([hit["text"] for hit in hits[:2]], ["가 학사 일정 요구사항", "가 학사 성적 요구사항 정의"])
        self.assertTrue(all(isinstance(hit[key], float) for hit in hits for key in ("score", "rerank_score", "rrf_score")))

    def test_default_keeps_filters(self):
        with patch.object(retrieval, "_cross_encoder_scores", side_effect=fake_cross_encoder):
            self.assertEqual({hit["doc_id"] for hit in self.search(5, {"발주 기관": "가"})}, {"bus"})
            self.assertEqual(self.search(5, {"발주 기관": "없음"}), [])

    def test_missing_packages_fall_back_to_lexical_with_warning(self):
        with patch.object(retrieval, "_cross_encoder_available", return_value=False), \
                patch.object(retrieval, "_cross_encoder_scores", side_effect=AssertionError("호출되면 안 됨")):
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(retrieval_options(), {**DEFAULTS, "rerank": "lexical"})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                hits = self.search(5)
        self.assertEqual(len(hits), 4)
        self.assertLessEqual([hit["doc_id"] for hit in hits].count("haksa"), 2)

    def test_explicit_cross_encoder_without_packages_does_not_fall_back(self):
        with patch.object(retrieval, "_cross_encoder_available", return_value=False), warnings.catch_warnings():
            warnings.simplefilter("error")  # 명시한 경우에는 대체 경고도 없어야 합니다.
            self.assertEqual(retrieval_options(rerank="cross-encoder")["rerank"], "cross-encoder")
            with patch.dict(os.environ, {"RETRIEVAL_RERANK": "cross-encoder"}):
                self.assertEqual(retrieval_options()["rerank"], "cross-encoder")

    def test_baseline_can_be_restored_by_arguments_and_environment(self):
        with patch.object(retrieval, "_cross_encoder_scores", side_effect=AssertionError("호출되면 안 됨")), \
                patch.object(retrieval, "_bm25_ranking", side_effect=AssertionError("호출되면 안 됨")):
            by_args = self.search(5, **BASELINE)
            with patch.dict(os.environ, {"RETRIEVAL_RERANK": "none", "RETRIEVAL_HYBRID": "off",
                                         "RETRIEVAL_MAX_PER_DOC": "none"}):
                self.assertEqual(retrieval_options()["max_per_doc"], None)
                by_env = self.search(5)
        self.assertEqual(by_args, by_env)
        self.assertEqual([hit["doc_id"] for hit in by_args], ["haksa"] * 3 + ["bus", "water"])

    def test_max_per_doc_argument_and_environment(self):
        cases = [({}, {}, 2), ({}, {"RETRIEVAL_MAX_PER_DOC": "3"}, 3), ({}, {"RETRIEVAL_MAX_PER_DOC": " Off "}, None),
                 ({}, {"RETRIEVAL_MAX_PER_DOC": ""}, 2), ({"max_per_doc": 1}, {"RETRIEVAL_MAX_PER_DOC": "none"}, 1),
                 ({"max_per_doc": "none"}, {"RETRIEVAL_MAX_PER_DOC": "3"}, None)]
        for kwargs, env, expected in cases:
            with self.subTest(kwargs=kwargs, env=env), patch.dict(os.environ, env):
                self.assertEqual(retrieval_options(**kwargs)["max_per_doc"], expected)
        for kwargs, env in (({"max_per_doc": 0}, {}), ({}, {"RETRIEVAL_MAX_PER_DOC": "0"}),
                            ({}, {"RETRIEVAL_MAX_PER_DOC": "many"}), ({"max_per_doc": True}, {})):
            with self.subTest(kwargs=kwargs, env=env), patch.dict(os.environ, env), self.assertRaises(ValueError):
                retrieval_options(**kwargs)


if __name__ == "__main__":
    unittest.main()
