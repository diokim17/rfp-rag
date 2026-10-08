"""BM25 사업명·발주 기관 접두(bm25_prefix)를 외부 API 없이 검증합니다.

검색 기본값은 baseline으로 고정해 검증하며, 새 기본값(켬)은 test_retrieval_defaults.py에서 봅니다.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from embedding import build_index, load_index
import retrieval
from retrieval import retrieval_options, retrieve
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정
from test_retrieval_rerank import FakeClient, document

QUESTION = "버스정보시스템"


class Bm25PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        # 'bus'는 사업명에만, 'noise'는 본문에만 질문 단어가 있습니다.
        build_index([
            document("bus", "버스정보시스템 구축", "경기도 평택시", "가 정류장 안내 단말기 설치"),
            document("noise", "홈페이지 개편", "가나재단", "다 버스정보시스템 연계 검토"),
            document("haksa", "학사정보시스템 고도화", "한영대학", "나 학사 일정 안내"),
        ], cls.client, cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        patcher = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("RETRIEVAL_")},
                             clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def ranked_docs(self, prefix):
        everything = set(range(len(self.chunks)))
        return [self.chunks[i]["doc_id"] for i in retrieval._bm25_ranking(QUESTION, self.chunks, everything, 10, prefix)]

    def test_prefix_matches_project_name_missing_from_body(self):
        self.assertEqual(self.ranked_docs(False), ["noise"])
        # '학사정보시스템'도 '정보·시스템' 2-gram을 공유하므로 걸리지만 두 문서보다 아래입니다.
        ranked = self.ranked_docs(True)
        self.assertEqual(set(ranked[:2]), {"bus", "noise"})
        self.assertEqual(ranked[2:], ["haksa"])

    def test_returned_text_and_score_are_unchanged(self):
        hits = retrieve(QUESTION, self.client, self.index, self.chunks, self.config, 3, None,
                        hybrid=True, rerank="none", max_per_doc="none", bm25_prefix=True)
        by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        for hit in hits:
            self.assertEqual(hit["text"], by_id[hit["chunk_id"]]["text"])
            self.assertNotIn("버스정보시스템 구축", hit["text"])
            self.assertIsInstance(hit["score"], float)

    def test_option_reaches_bm25_ranking(self):
        for value in (True, False):
            with self.subTest(value=value), patch.object(retrieval, "_bm25_ranking",
                                                         wraps=retrieval._bm25_ranking) as spy:
                retrieve(QUESTION, self.client, self.index, self.chunks, self.config, 3, None,
                         hybrid=True, rerank="none", bm25_prefix=value)
                self.assertEqual(spy.call_args.args[4], value)

    def test_options_argument_environment_and_hybrid_off(self):
        self.assertIsNone(retrieval_options(hybrid=False, bm25_prefix=True)["bm25_prefix"])
        self.assertFalse(retrieval_options(hybrid=True)["bm25_prefix"])  # baseline 고정값
        with patch.object(retrieval, "DEFAULT_BM25_PREFIX", True):
            self.assertTrue(retrieval_options(hybrid=True)["bm25_prefix"])
            self.assertFalse(retrieval_options(hybrid=True, bm25_prefix=False)["bm25_prefix"])
            with patch.dict(os.environ, {"RETRIEVAL_BM25_PREFIX": "off"}):
                self.assertFalse(retrieval_options(hybrid=True)["bm25_prefix"])
                self.assertTrue(retrieval_options(hybrid=True, bm25_prefix=True)["bm25_prefix"])
        with patch.dict(os.environ, {"RETRIEVAL_BM25_PREFIX": "maybe"}), self.assertRaises(ValueError):
            retrieval_options(hybrid=True)

    def test_body_and_prefix_indexes_are_cached_together(self):
        body = retrieval._bm25_index(self.chunks, False)
        prefixed = retrieval._bm25_index(self.chunks, True)
        self.assertIsNot(body, prefixed)
        self.assertIs(retrieval._bm25_index(self.chunks, False), body)
        self.assertIs(retrieval._bm25_index(self.chunks, True), prefixed)
        other = [dict(chunk) for chunk in self.chunks]
        retrieval._bm25_index(other, False)
        self.assertTrue(all(value["chunks"] is other for value in retrieval._bm25_indexes.values()))


if __name__ == "__main__":
    unittest.main()
