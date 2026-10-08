"""같은 문서 안에서 청크 다시 고르기(within_doc)를 외부 API 없이 검증합니다.

검색 기본값은 baseline으로 고정해 검증합니다(within_doc 기본은 off).
"""

import os
import unittest
from unittest.mock import patch

import faiss

from embedding import embed_texts
import retrieval
from retrieval import _residual_question, retrieval_options, retrieve
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정
from test_retrieval_rerank import FakeClient

QUESTION = "고려대학교 포털 구축 사업의 발표 시간은?"
PROJECT = {"사업명": "고려대학교 포털 구축 사업", "발주 기관": "고려대학교"}


def chunk(chunk_id, text, metadata):
    doc_id = chunk_id.split(":")[0]
    return {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "metadata": {**metadata, "filename": f"{doc_id}.hwp"}}


class WithinDocTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = FakeClient()
        # FakeClient는 첫 글자로 벡터를 정합니다. 질문('고')과 kp:0('고')은 가깝고 kp:1('가')은 멉니다.
        # 발표 시간은 kp:1에만 있어, 사업명이 겹치는 kp:0이 1단계에서 먼저 뽑힙니다.
        cls.chunks = [
            chunk("kp:0", "고려대학교 포털 구축 사업 개요와 추진 배경", PROJECT),
            chunk("kp:1", "가 제안서 발표 시간은 30분, 질의응답 10분", PROJECT),
            chunk("other:0", "나 다른 사업의 일정", {"사업명": "버스정보시스템 구축", "발주 기관": "평택시"}),
        ]
        vectors = embed_texts([c["text"] for c in cls.chunks], cls.client, "test-embedding")
        cls.index = faiss.IndexFlatIP(vectors.shape[1])
        cls.index.add(vectors)
        cls.config = {"embedding_model": "test-embedding"}

    def setUp(self):
        patcher = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("RETRIEVAL_")},
                             clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def search(self, top_k=1, **options):
        return retrieve(QUESTION, self.client, self.index, self.chunks, self.config, top_k, None, **options)

    def test_residual_question_drops_project_words(self):
        self.assertEqual(_residual_question(QUESTION, PROJECT), "발표 시간은?")
        self.assertEqual(_residual_question("고려대학교 포털 구축 사업", PROJECT), "고려대학교 포털 구축 사업")
        self.assertEqual(_residual_question(QUESTION, {}), QUESTION)

    def test_off_by_default_and_no_extra_api_call(self):
        calls = self.client.calls
        hits = self.search()
        self.assertEqual([hit["chunk_id"] for hit in hits], ["kp:0"])
        self.assertEqual(self.client.calls - calls, 1)
        self.assertTrue(all("within_doc_rank" not in hit for hit in hits))

    def test_reselects_answer_chunk_inside_the_same_document(self):
        for mode in ("residual", "residual+full"):
            with self.subTest(mode=mode):
                calls = self.client.calls
                hits = self.search(within_doc=mode)
                self.assertEqual([hit["chunk_id"] for hit in hits], ["kp:1"])
                self.assertEqual(self.client.calls - calls, 2)  # 1단계 1회 + 재선택 1회(묶어서)
                expected = float(embed_texts([QUESTION], FakeClient(), "x")[0] @ self.index.reconstruct(1))
                self.assertAlmostEqual(hits[0]["score"], expected, places=6)  # score는 여전히 질문과의 코사인
                self.assertEqual(hits[0]["within_doc_rank"], 1)
                self.assertEqual(hits[0]["text"], self.chunks[1]["text"])

    def test_document_slots_and_order_are_kept(self):
        base = self.search(top_k=3, max_per_doc="none")
        hits = self.search(top_k=3, max_per_doc="none", within_doc="residual")
        self.assertEqual([hit["doc_id"] for hit in hits], [hit["doc_id"] for hit in base])
        self.assertEqual(len({hit["chunk_id"] for hit in hits}), len(hits))  # 같은 청크가 두 번 나오지 않음
        kp = [hit for hit in hits if hit["doc_id"] == "kp"]
        self.assertEqual([hit["within_doc_rank"] for hit in kp], [1, 2])
        self.assertEqual(kp[0]["chunk_id"], "kp:1")

    def test_unchanged_chunk_keeps_original_fields(self):
        base = {hit["chunk_id"]: hit for hit in self.search(top_k=3, hybrid=True, max_per_doc="none")}
        hits = self.search(top_k=3, hybrid=True, max_per_doc="none", within_doc="residual+full")
        other = next(hit for hit in hits if hit["doc_id"] == "other")
        self.assertEqual({k: v for k, v in other.items() if k != "within_doc_rank"}, base["other:0"])
        self.assertIn("rrf_score", other)

    def test_options_argument_environment_and_invalid(self):
        self.assertEqual(retrieval_options()["within_doc"], "off")
        self.assertEqual(retrieval_options(within_doc="residual")["within_doc"], "residual")
        with patch.dict(os.environ, {"RETRIEVAL_WITHIN_DOC": "residual+full"}):
            self.assertEqual(retrieval_options()["within_doc"], "residual+full")
            self.assertEqual(retrieval_options(within_doc="off")["within_doc"], "off")
        with self.assertRaises(ValueError):
            retrieval_options(within_doc="all")

    def test_empty_result_skips_reselection(self):
        calls = self.client.calls
        hits = retrieve(QUESTION, self.client, self.index, self.chunks, self.config, 3, {"발주 기관": "없는기관"},
                        within_doc="residual")
        self.assertEqual(hits, [])
        self.assertEqual(self.client.calls, calls)

    def test_member_cache_follows_chunk_list(self):
        retrieval._members(self.chunks)
        other = [dict(c) for c in self.chunks[:1]]
        self.assertEqual(retrieval._members(other), {"kp": [0]})
        self.assertEqual(retrieval._members(self.chunks)["kp"], [0, 1])


if __name__ == "__main__":
    unittest.main()
