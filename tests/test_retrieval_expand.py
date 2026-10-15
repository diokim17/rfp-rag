"""재정렬 후보 확장(expand)을 외부 API·모델 없이 검증합니다. cross-encoder 점수는 가짜 함수로 대체합니다."""

import os
import unittest
from unittest.mock import patch

import faiss

from b_embedding import embed_texts
import b_retrieval as retrieval
from b_retrieval import retrieval_options, retrieve
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정
from test_retrieval_rerank import FakeClient

QUESTION = "고려대학교 포털 구축 사업의 발표 시간은?"
PROJECT = {"사업명": "고려대학교 포털 구축 사업", "발주 기관": "고려대학교"}
OTHER = {"사업명": "버스정보시스템 구축", "발주 기관": "평택시"}


def chunk(chunk_id, text, metadata):
    doc_id = chunk_id.split(":")[0]
    return {"chunk_id": chunk_id, "doc_id": doc_id, "text": text, "metadata": {**metadata, "filename": f"{doc_id}.hwp"}}


def fake_cross_encoder(question, passages, model_name):
    """'발표'가 든 문단을 가장 높게 보는 결정적인 점수."""
    return [2. if "발표" in passage.split("\n", 1)[1] else 1. for passage in passages]


class ExpandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = FakeClient()
        # kp:0·kp:2는 질문('고')과 벡터가 가깝고 kp:1(발표 시간)은 멉니다. 후보가 적으면 kp:1이 빠집니다.
        cls.chunks = [
            chunk("kp:0", "고려대학교 포털 구축 사업 개요", PROJECT),
            chunk("kp:1", "가 제안서 발표 시간은 30분, 질의응답 10분", PROJECT),
            chunk("kp:2", "고려대학교 포털 구축 사업 추진 배경", PROJECT),
            chunk("other:0", "고 다른 사업의 발표 일정", OTHER),
            chunk("third:0", "고 세 번째 사업 발표 시간 안내", {"사업명": "세 번째 사업", "발주 기관": "기관"}),
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
        scores = patch.object(retrieval, "_cross_encoder_scores", side_effect=fake_cross_encoder)
        self.cross_encoder = scores.start()
        self.addCleanup(scores.stop)

    def search(self, top_k=1, **options):
        return retrieve(QUESTION, self.client, self.index, self.chunks, self.config, top_k, None,
                        rerank="cross-encoder", candidates=1, max_per_doc="none", **options)

    def test_off_by_default_keeps_candidates(self):
        self.assertEqual([hit["chunk_id"] for hit in self.search()], ["kp:0"])
        self.assertEqual(len(self.cross_encoder.call_args.args[1]), 1)

    def test_expanded_answer_chunk_is_ranked_by_reranker(self):
        calls = self.client.calls
        hits = self.search(expand=1)
        self.assertEqual(self.client.calls - calls, 1)  # API 호출은 늘지 않음
        self.assertEqual([hit["chunk_id"] for hit in hits], ["kp:1"])
        self.assertTrue(hits[0]["expanded"])
        expected = float(embed_texts([QUESTION], FakeClient(), "x")[0] @ self.index.reconstruct(1))
        self.assertAlmostEqual(hits[0]["score"], expected, places=6)  # score는 여전히 질문과의 코사인
        self.assertNotIn("rrf_score", hits[0])

    def test_hybrid_path_also_expands(self):
        hits = self.search(expand=1, hybrid=True)
        self.assertEqual([hit["chunk_id"] for hit in hits], ["kp:1"])

    def test_only_unseen_chunks_from_first_two_documents(self):
        # 후보 3개 = kp:0, kp:2, other:0 (문서 2개). 확장은 이 두 문서에서만, 이미 후보인 청크는 빼고 더합니다.
        retrieve(QUESTION, self.client, self.index, self.chunks, self.config, 1, None,
                 rerank="cross-encoder", candidates=3, max_per_doc="none", expand=5)
        passages = self.cross_encoder.call_args.args[1]
        self.assertEqual(len(passages), len(set(passages)))  # 같은 청크를 두 번 넣지 않음
        self.assertEqual(len(passages), 4)  # 원래 후보 3 + kp:1 (other에는 남은 청크가 없음)
        self.assertTrue(any("발표 시간은 30분" in p for p in passages))
        self.assertFalse(any(p.startswith("세 번째 사업") for p in passages))  # 세 번째 문서는 확장하지 않음

    def test_respects_filters(self):
        hits = retrieve(QUESTION, self.client, self.index, self.chunks, self.config, 5, {"발주 기관": "고려대학교"},
                        rerank="cross-encoder", candidates=1, max_per_doc="none", expand=5)
        self.assertEqual({hit["doc_id"] for hit in hits}, {"kp"})

    def test_options(self):
        self.assertIsNone(retrieval_options(rerank="none", expand=5)["expand"])
        self.assertEqual(retrieval_options(rerank="lexical")["expand"], 0)
        self.assertEqual(retrieval_options(rerank="lexical", expand=10)["expand"], 10)
        with patch.dict(os.environ, {"RETRIEVAL_EXPAND": "7"}):
            self.assertEqual(retrieval_options(rerank="lexical")["expand"], 7)
            self.assertEqual(retrieval_options(rerank="lexical", expand=0)["expand"], 0)
        for bad in (-1, True, "many"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                retrieval_options(rerank="lexical", expand=bad)


if __name__ == "__main__":
    unittest.main()


class NextChunkTests(unittest.TestCase):
    """1위 청크의 바로 다음 청크(같은 문서)를 덧붙이고, 이미 있거나 문서 끝이면 그대로 둡니다."""

    def test_with_next_chunk(self):
        chunks = [chunk("a:0", "목록 앞", PROJECT), chunk("a:1", "목록 뒤", PROJECT), chunk("b:0", "다른 문서", OTHER)]
        top = [{**chunks[0], "score": .9}]
        self.assertEqual([(h["chunk_id"], h.get("neighbor")) for h in retrieval.with_next_chunk(top, chunks)],
                         [("a:0", None), ("a:1", True)])
        both = [{**chunks[0], "score": .9}, {**chunks[1], "score": .8}]
        self.assertEqual(retrieval.with_next_chunk(both, chunks), both)
        last = [{**chunks[1], "score": .9}]
        self.assertEqual(retrieval.with_next_chunk(last, chunks), last)
        self.assertEqual(retrieval.with_next_chunk([], chunks), [])


class AspectTests(unittest.TestCase):
    """여러 문장 질문: 찾은 청크와 덜 겹치는 문장(묻는 항목)으로 1위 문서의 청크 순서를 다시 정합니다."""

    QUESTION = "다음 내용의 사업에서 입찰 참가 자격은 무엇인가요? 학사포털 파일 암호화 모듈 적용"

    def test_aspect_sentence(self):
        self.assertEqual(retrieval._aspect_sentence(self.QUESTION, "학사포털 파일 암호화 모듈 적용 배경"),
                         "다음 내용의 사업에서 입찰 참가 자격은 무엇인가요?")
        self.assertIsNone(retrieval._aspect_sentence("입찰 참가 자격은 무엇인가요?", "아무 내용"))

    def test_aspect_slot_reorders_top_doc_only(self):
        ranked = [chunk("a:0", "학사포털 파일 암호화 모듈 적용 배경", PROJECT),
                  chunk("a:1", "학사포털 암호화 추진 목적", PROJECT),
                  chunk("b:0", "다른 사업의 입찰 참가 자격", OTHER),
                  chunk("a:2", "입찰 참가 자격: 소프트웨어사업자", PROJECT)]
        scores = lambda question, passages, model: [2. if "자격" in p else 1. for p in passages]
        with patch.object(retrieval, "_cross_encoder_scores", side_effect=scores):
            result = retrieval._aspect_slot(self.QUESTION, ranked, "cross-encoder")
            self.assertEqual([h["chunk_id"] for h in result], ["a:2", "a:0", "a:1", "b:0"])
            single = retrieval._aspect_slot("입찰 참가 자격은?", ranked, "cross-encoder")
        self.assertEqual(single, ranked)
