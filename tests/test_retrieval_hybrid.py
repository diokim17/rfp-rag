"""하이브리드 검색(BM25 + RRF) 옵션이 baseline에서 꺼져 있고, 켜면 실제로 동작하는지 외부 API 없이 검증합니다.

검색 기본값은 baseline으로 고정해 검증하며, 새 기본값은 test_retrieval_defaults.py에서 봅니다.
"""

import os
import re
import tempfile
import unittest
from unittest.mock import patch

from embedding import build_index, load_index
import retrieval
from retrieval import retrieval_options, retrieve
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정
from test_retrieval_rerank import FakeClient, document

QUESTION = "학사정보시스템 고도화 사업의 요구사항"
ENV_KEYS = ("RETRIEVAL_RERANK", "RETRIEVAL_CANDIDATES", "RETRIEVAL_HYBRID", "RETRIEVAL_HYBRID_VECTOR_K",
            "RETRIEVAL_HYBRID_BM25_K", "RETRIEVAL_RRF_K", "RETRIEVAL_REWRITE", "RETRIEVAL_REWRITE_MODEL",
            "RETRIEVAL_MAX_PER_DOC", "RETRIEVAL_BM25_PREFIX")


class HybridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        # FakeClient는 본문 첫 글자로 코사인 순서를 정합니다(가 > 나 > 다 > 그 외).
        # "keyword"는 코사인이 가장 낮지만 본문 단어가 질문과 가장 많이 겹치는 청크입니다.
        build_index([
            document("bus", "버스정보시스템 구축", "가", "가 버스 노선 안내와 제안 안내"),
            document("bus", "버스정보시스템 구축", "가", "가 버스 정류장 정보와 제출 서류"),
            document("water", "용수공급 타당성조사", "나", "나 용수 공급 계획과 제안 안내"),
            document("haksa", "학사정보시스템 고도화", "나", "다 학사 일정과 제안 안내"),
            document("keyword", "통합 학사행정 개선", "다", "기타 학사정보시스템 고도화 사업 요구사항 정의"),
        ], cls.client, cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        patcher = patch.dict(os.environ, {key: value for key, value in os.environ.items() if key not in ENV_KEYS},
                             clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def search(self, top_k=3, filters=None, **options):
        return retrieve(QUESTION, self.client, self.index, self.chunks, self.config, top_k, filters, **options)

    def docs(self, hits):
        return [hit["doc_id"] for hit in hits]

    def test_defaults_are_off_and_match_baseline(self):
        options = retrieval_options()
        self.assertEqual(options, {"rerank": "none", "candidates": None, "max_per_doc": None, "hybrid": False,
                                   "hybrid_vector_k": None, "hybrid_bm25_k": None, "rrf_k": None,
                                   "rewrite": "off", "rewrite_model": None, "bm25_prefix": None})
        with patch.object(retrieval, "_bm25_ranking", side_effect=AssertionError("BM25가 호출되면 안 됨")):
            default = self.search(3)
            off = self.search(3, hybrid=False, rewrite="off")
        self.assertEqual(default, off)
        self.assertEqual(self.docs(default), ["bus", "bus", "water"])
        self.assertTrue(all("rrf_score" not in hit for hit in default))

    def test_hybrid_option_really_runs_bm25_and_adds_keyword_match(self):
        spy = patch.object(retrieval, "_bm25_ranking", wraps=retrieval._bm25_ranking)
        with spy as bm25:
            hits = self.search(3, hybrid=True, hybrid_vector_k=2, hybrid_bm25_k=1)
        bm25.assert_called_once()
        # 코사인만으로는 꼴찌인 keyword 청크가 BM25 후보로 들어와 RRF 순위 상위에 섭니다.
        self.assertIn("keyword", self.docs(hits))
        self.assertNotIn("keyword", self.docs(self.search(3)))
        self.assertTrue(all(isinstance(hit["rrf_score"], float) for hit in hits))
        self.assertEqual(retrieval_options(hybrid=True)["hybrid_bm25_k"], 100)

    def test_environment_turns_hybrid_on_and_argument_wins(self):
        with patch.dict(os.environ, {"RETRIEVAL_HYBRID": "on", "RETRIEVAL_HYBRID_BM25_K": "1",
                                     "RETRIEVAL_HYBRID_VECTOR_K": "2"}):
            self.assertTrue(retrieval_options()["hybrid"])
            self.assertIn("keyword", self.docs(self.search(3)))
            self.assertFalse(retrieval_options(hybrid=False)["hybrid"])
            self.assertNotIn("keyword", self.docs(self.search(3, hybrid=False)))

    def test_docstring_environment_variables_change_options(self):
        """독스트링에 적힌 환경 변수가 실제로 옵션을 바꾸는지 확인합니다 (설명과 동작 불일치 방지)."""
        documented = set(re.findall(r"RETRIEVAL_[A-Z0-9_]+", retrieve.__doc__))
        self.assertTrue(documented >= {"RETRIEVAL_RERANK", "RETRIEVAL_CANDIDATES", "RETRIEVAL_HYBRID",
                                       "RETRIEVAL_HYBRID_VECTOR_K", "RETRIEVAL_HYBRID_BM25_K", "RETRIEVAL_RRF_K",
                                       "RETRIEVAL_REWRITE", "RETRIEVAL_REWRITE_MODEL", "RETRIEVAL_MAX_PER_DOC",
                                       "RETRIEVAL_BM25_PREFIX"})
        base = {"RETRIEVAL_RERANK": "lexical", "RETRIEVAL_HYBRID": "on", "RETRIEVAL_REWRITE": "only"}
        with patch.dict(os.environ, base):
            before = retrieval_options()
        changes = {"RETRIEVAL_RERANK": "none", "RETRIEVAL_CANDIDATES": "7", "RETRIEVAL_HYBRID": "off",
                   "RETRIEVAL_HYBRID_VECTOR_K": "3", "RETRIEVAL_HYBRID_BM25_K": "4", "RETRIEVAL_RRF_K": "10",
                   "RETRIEVAL_REWRITE": "both", "RETRIEVAL_REWRITE_MODEL": "gpt-5-nano", "RETRIEVAL_MAX_PER_DOC": "3",
                   "RETRIEVAL_BM25_PREFIX": "on"}
        for name in documented:
            with self.subTest(name=name), patch.dict(os.environ, {**base, name: changes[name]}):
                self.assertNotEqual(retrieval_options(), before)

    def test_score_stays_cosine_and_filters_hold(self):
        cosine = {hit["chunk_id"]: hit["score"] for hit in self.search(len(self.chunks))}
        for rerank in ("none", "lexical"):
            hits = self.search(5, hybrid=True, rerank=rerank)
            self.assertTrue(all(hit["score"] == cosine[hit["chunk_id"]] for hit in hits), rerank)
            filtered = self.search(5, {"발주 기관": "나"}, hybrid=True, rerank=rerank)
            self.assertEqual(sorted(set(self.docs(filtered))), ["haksa", "water"], rerank)
        self.assertEqual(self.search(5, {"발주 기관": "없음"}, hybrid=True), [])

    def test_lexical_rerank_applies_to_fused_candidates(self):
        with patch.object(retrieval, "_rerank", wraps=retrieval._rerank) as rerank:
            hits = self.search(2, hybrid=True, rerank="lexical", candidates=4, hybrid_vector_k=2, hybrid_bm25_k=2)
        rerank.assert_called_once()
        reranked = rerank.call_args.args[1]
        self.assertEqual(len(reranked), 4)
        self.assertIn("keyword", self.docs(reranked))
        self.assertTrue(all("rerank_score" in hit and "rrf_score" in hit for hit in hits))

    def test_max_per_doc_with_hybrid(self):
        for rerank in ("none", "lexical"):
            hits = self.search(4, hybrid=True, rerank=rerank, max_per_doc=1)
            self.assertEqual(len(self.docs(hits)), len(set(self.docs(hits))), rerank)
        self.assertEqual(self.docs(self.search(2, hybrid=True, hybrid_vector_k=2, hybrid_bm25_k=1)).count("bus"), 1)

    def test_rrf_and_bm25_building_blocks(self):
        self.assertEqual(retrieval._rrf([[1, 2], [2, 3]], 60)[0][0], 2)  # 두 목록에 모두 있으면 위로
        self.assertEqual([i for i, _ in retrieval._rrf([[1], [2]], 60)], [1, 2])  # 동점은 먼저 나온 순서
        ranking = retrieval._bm25_ranking(QUESTION, self.chunks, set(range(len(self.chunks))), 10)
        self.assertEqual(self.chunks[ranking[0]]["doc_id"], "keyword")
        self.assertNotIn(0, ranking)  # 겹치는 단어가 없는 청크(버스 노선)는 BM25 후보가 아님
        self.assertEqual(retrieval._bm25_ranking(QUESTION, self.chunks, {0}, 10), [])

    def test_invalid_options(self):
        for options in ({"hybrid": "maybe"}, {"hybrid": True, "rrf_k": 0}, {"hybrid": True, "hybrid_bm25_k": 0},
                        {"hybrid": True, "hybrid_vector_k": 0}, {"rewrite": "always"}, {"max_per_doc": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.search(**options)


if __name__ == "__main__":
    unittest.main()
