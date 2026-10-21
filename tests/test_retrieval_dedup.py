"""dedup: 본문이 같은 문서·청크가 최종 결과 자리를 겹쳐 차지하지 않는지 오프라인으로 검증합니다."""

import os
import tempfile
import unittest
from unittest.mock import patch

from pipelines.api.api_embedding import build_index, load_index
from pipelines.api.api_retrieval import retrieval_options, retrieve
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정
from test_retrieval_rerank import FakeClient, document

# 10글자씩 잘라 청크 하나가 한 조각이 되게 합니다. 첫 글자가 질문과의 가까움을 정합니다(가 > 나 > 다).
PIECES = {"a": "가111111111", "b": "가222222222", "c": "나333333333", "d": "다444444444"}
SAME_TEXT = PIECES["a"] + PIECES["b"] + PIECES["c"]


class DedupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "fixed"}):
            build_index([
                document("orig", "통합정보시스템 고도화", "가", SAME_TEXT),
                document("copy", "통합정보시스템 고도화", "나", SAME_TEXT),  # 같은 공고가 다른 파일명으로 한 번 더
                document("other", "버스정보시스템 구축", "다", PIECES["d"] + PIECES["a"]),  # 뒤 조각은 orig와 같은 서식 문구
            ], cls.client, cls.directory.name, model="test-embedding", chunk_size=10, chunk_overlap=0)
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def search(self, **options):
        hits = retrieve("학사 요구사항", self.client, self.index, self.chunks, self.config, top_k=5, **options)
        return [(hit["doc_id"], hit["text"]) for hit in hits]

    def test_off_keeps_duplicates(self):
        hits = self.search(max_per_doc=2)
        self.assertEqual(len(hits), 5)
        self.assertEqual({doc_id for doc_id, _ in hits}, {"orig", "copy", "other"})

    def test_same_documents_share_max_per_doc_and_same_text_is_skipped(self):
        for options in ({}, {"rerank": "lexical", "hybrid": True}):  # 코사인 경로와 하이브리드+리랭킹 경로
            with self.subTest(**options):
                hits = self.search(max_per_doc=2, dedup=True, **options)
                # 점수가 같은 청크의 순서는 FAISS가 정하므로 어느 쪽이 남는지가 아니라 겹치지 않는지만 봅니다.
                texts = [text for _, text in hits]
                self.assertEqual(len(texts), len(set(texts)))  # 같은 본문은 한 번만
                self.assertEqual(sum(doc_id in ("orig", "copy") for doc_id, _ in hits), 2)  # orig·copy 합쳐 상한 2
                self.assertIn(("other", PIECES["d"]), hits)

    def test_without_max_per_doc_only_same_text_is_skipped(self):
        texts = [text for _, text in self.search(max_per_doc="none", dedup=True)]
        self.assertEqual(sorted(texts), sorted(PIECES.values()))

    def test_within_doc_reselect_keeps_dedup(self):
        # 문서 안 재선택은 자리마다 청크를 다시 고르므로, 다른 문서와 같은 본문(other의 뒤 조각)을 다시 고르면 안 됩니다.
        for mode in ("residual", "residual+full"):
            with self.subTest(within_doc=mode):
                hits = self.search(max_per_doc=2, dedup=True, within_doc=mode)
                texts = [text for _, text in hits]
                self.assertEqual(len(texts), len(set(texts)))
                self.assertEqual(sum(doc_id in ("orig", "copy") for doc_id, _ in hits), 2)

    def test_option_default_off_and_env(self):
        with patch.dict(os.environ, {"RETRIEVAL_DEDUP": ""}):
            self.assertFalse(retrieval_options()["dedup"])
        with patch.dict(os.environ, {"RETRIEVAL_DEDUP": "on"}):
            self.assertTrue(retrieval_options()["dedup"])


if __name__ == "__main__":
    unittest.main()
