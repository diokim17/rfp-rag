"""외부 API·모델 다운로드 없이 리랭킹의 기본값, 필터 유지, 점수 계약과 모듈 연결을 검증합니다."""

import json
import math
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embedding import build_index, load_index
from evaluation import evaluate
from generation import generate_answer
from observability import Trace
import retrieval
from retrieval import retrieve
from test_observability import FakeLangfuse


class FakeClient:
    """본문 첫 글자로 질문과의 코사인 순서(가 > 나 > 다 > 그 외)를 정합니다. 사업명은 임베딩에 반영되지 않습니다."""

    CLOSENESS = {"학": 1., "가": .9, "나": .8, "다": .7}  # '학'은 질문의 첫 글자

    def __init__(self):
        self.embeddings = SimpleNamespace(create=self.embed)
        self.responses = SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text="요구사항입니다. [1]"))
        self.calls = 0

    def embed(self, model, input):
        self.calls += 1
        closeness = [self.CLOSENESS.get(text[0], 0.) for text in input]
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[c, 1. - c]) for i, c in enumerate(closeness)])


def document(doc_id, name, agency, text):
    return {"doc_id": doc_id, "text": text,
            "metadata": {"사업명": name, "발주 기관": agency, "filename": f"{doc_id}.hwp"}}


class RerankTests(unittest.TestCase):
    QUESTION = "학사정보시스템 고도화 사업의 요구사항"

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        # 본문은 문서마다 비슷하고 사업명은 메타데이터에만 있는 실제 RFP 상황을 축소한 자료입니다.
        build_index([
            document("bus", "버스정보시스템 구축", "가", "가 요구사항 목록과 제안 안내"),
            document("haksa", "학사정보시스템 고도화", "나", "다 요구사항 목록과 제안 안내"),
            document("water", "용수공급 타당성조사", "나", "나 요구사항 목록과 제안 안내"),
            document("etc", "학사정보시스템 고도화 감리", "다", "일정과 예산"),
        ], cls.client, cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def search(self, top_k=5, filters=None, **options):
        return retrieve(self.QUESTION, self.client, self.index, self.chunks, self.config, top_k, filters, **options)

    def test_default_is_baseline_without_rerank_fields(self):
        with patch.dict(os.environ, {}, clear=True):
            default = self.search(3)
        self.assertEqual(default, self.search(3, rerank="none"))
        self.assertEqual([hit["doc_id"] for hit in default], ["bus", "water", "haksa"])
        self.assertEqual([hit["score"] for hit in default], sorted((hit["score"] for hit in default), reverse=True))
        for hit in default:
            self.assertEqual(set(hit), {"chunk_id", "doc_id", "text", "metadata", "score"})

    def test_lexical_rerank_promotes_matching_project_and_keeps_contract(self):
        baseline = {hit["chunk_id"]: hit for hit in self.search(4)}
        hits = self.search(2, rerank="lexical")
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0]["doc_id"], "haksa")
        for hit in hits:
            # score는 코사인 유사도 그대로이고 Chunk 필드도 변하지 않습니다.
            self.assertEqual({k: v for k, v in hit.items() if k != "rerank_score"}, baseline[hit["chunk_id"]])
            self.assertIs(type(hit["score"]), float)
            self.assertIs(type(hit["rerank_score"]), float)
            self.assertTrue(math.isfinite(hit["rerank_score"]))
        self.assertGreaterEqual(hits[0]["rerank_score"], hits[1]["rerank_score"])
        self.assertEqual(json.loads(json.dumps(hits, ensure_ascii=False)), hits)

    def test_rerank_never_returns_documents_outside_filter(self):
        for mode in ("none", "lexical"):
            hits = self.search(5, {"발주 기관": "가"}, rerank=mode)
            self.assertEqual([hit["doc_id"] for hit in hits], ["bus"], mode)
        hits = self.search(5, {"발주 기관": "나"}, rerank="lexical")
        self.assertEqual({hit["doc_id"] for hit in hits}, {"haksa", "water"})
        self.assertTrue(all(hit["metadata"]["발주 기관"] == "나" for hit in hits))
        calls = self.client.calls
        self.assertEqual(self.search(5, {"발주 기관": "없음"}, rerank="lexical"), [])
        self.assertEqual(self.search(5, {"없는 키": "가"}, rerank="lexical"), [])
        self.assertEqual(self.client.calls, calls)  # 조건에 맞는 청크가 없으면 API를 호출하지 않습니다.

    def test_candidate_pool_bounds(self):
        # 후보가 코사인 상위 2개뿐이면 그 밖의 문서는 리랭킹으로도 올라오지 않습니다.
        self.assertEqual({hit["doc_id"] for hit in self.search(2, rerank="lexical", candidates=2)}, {"bus", "water"})
        # 후보 수가 top_k보다 작아도 top_k개를 반환하고, 청크 수보다 커도 동작합니다.
        self.assertEqual(len(self.search(3, rerank="lexical", candidates=1)), 3)
        self.assertEqual(len(self.search(10, rerank="lexical", candidates=1000)), len(self.chunks))
        self.assertEqual(len(self.search(1, rerank="lexical")), 1)

    def test_settings_from_environment_and_validation(self):
        with patch.dict(os.environ, {"RETRIEVAL_RERANK": " Lexical ", "RETRIEVAL_CANDIDATES": "4"}, clear=True):
            self.assertEqual(self.search(1)[0]["doc_id"], "haksa")
            self.assertNotIn("rerank_score", self.search(1, rerank="none")[0])  # 인자가 환경 변수보다 우선
        with patch.dict(os.environ, {"RETRIEVAL_RERANK": "lexical", "RETRIEVAL_CANDIDATES": "2"}, clear=True):
            self.assertNotEqual(self.search(1)[0]["doc_id"], "haksa")
        with patch.dict(os.environ, {"RETRIEVAL_RERANK": "unknown"}, clear=True):
            with self.assertRaises(ValueError):
                self.search()
        for options in ({"rerank": "bm25"}, {"rerank": "lexical", "candidates": 0}):
            with self.assertRaises(ValueError):
                self.search(**options)
        with self.assertRaises(ValueError):
            retrieve(" ", self.client, self.index, self.chunks, self.config, rerank="lexical")

    def test_cross_encoder_scores_decide_order_and_are_validated(self):
        def scorer(question, passages, model_name):
            self.assertTrue(all(passage.count("\n") == 1 for passage in passages))
            return [3. if passage.startswith("용수공급") else 0. for passage in passages]

        with patch.object(retrieval, "_cross_encoder_scores", scorer):
            hits = self.search(4, rerank="cross-encoder")
        self.assertEqual(hits[0]["doc_id"], "water")
        self.assertEqual(hits[0]["rerank_score"], 3.)
        # 점수가 같은 후보는 코사인 유사도 순서를 유지합니다.
        self.assertEqual([hit["doc_id"] for hit in hits[1:]], [hit["doc_id"] for hit in self.search(4) if hit["doc_id"] != "water"])
        for bad in (lambda q, p, m: [1.], lambda q, p, m: [float("nan")] * len(p)):
            with patch.object(retrieval, "_cross_encoder_scores", bad), self.assertRaises(ValueError):
                self.search(4, rerank="cross-encoder")

    def test_cross_encoder_without_optional_packages_fails_clearly(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            with self.assertRaisesRegex(RuntimeError, "torch"):
                self.search(rerank="cross-encoder")
            self.assertEqual(len(self.search(2)), 2)  # baseline 경로는 선택 패키지 없이 동작

    def test_reranked_hits_connect_to_generation_evaluation_and_tracing(self):
        langfuse = FakeLangfuse()
        with Trace({}, langfuse).run("test"):
            hits = self.search(2, rerank="lexical")
        names = [record["name"] for record in langfuse.records]
        self.assertIn("rerank-lexical", names)
        results = next(record for record in langfuse.records if record["name"] == "retrieval-results")
        self.assertEqual(results["output"], [{"doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "score": h["score"]}
                                             for h in hits])
        # 질문·본문·사업명은 추적 기록에 남지 않습니다.
        for private in ("학사정보시스템", "요구사항", "제안 안내"):
            self.assertNotIn(private, str(langfuse.records))

        answer = generate_answer(self.QUESTION, hits, self.client)
        self.assertEqual([source["citation"] for source in answer["sources"]], [1, 2])
        self.assertEqual(answer["sources"][0]["doc_id"], "haksa")
        report = evaluate([{"question": self.QUESTION, "expected_doc_ids": ["haksa"]}], lambda q, f: answer)
        self.assertEqual(report["summary"]["recall_at_k"], 1.)
        json.dumps(report, ensure_ascii=False)

    def test_max_per_doc_limits_chunks_from_one_document(self):
        with tempfile.TemporaryDirectory() as directory:
            build_index([
                document("bus", "버스정보시스템 구축", "가", "가 요구사항 목록과 제안 안내"),
                document("bus", "버스정보시스템 구축", "가", "가 요구사항 상세와 제출 서류"),
                document("water", "용수공급 타당성조사", "나", "나 요구사항 목록과 제안 안내"),
                document("haksa", "학사정보시스템 고도화", "나", "다 요구사항 목록과 제안 안내"),
            ], self.client, directory, model="test-embedding")
            index, chunks, config = load_index(directory)
            search = lambda top_k, filters=None, **options: [hit["doc_id"] for hit in retrieve(
                self.QUESTION, self.client, index, chunks, config, top_k, filters, **options)]
            self.assertEqual(search(3), ["bus", "bus", "water"])
            # 상한을 넘는 청크는 건너뛰고 다음 순위 문서로 top_k를 채웁니다.
            self.assertEqual(search(3, max_per_doc=1), ["bus", "water", "haksa"])
            self.assertEqual(search(3, max_per_doc=2), ["bus", "bus", "water"])
            self.assertEqual(search(3, {"발주 기관": "가"}, max_per_doc=1), ["bus"])
            # 리랭킹 뒤에도 상한을 지킵니다.
            self.assertEqual(sorted(search(4, rerank="lexical")), ["bus", "bus", "haksa", "water"])
            self.assertEqual(sorted(search(4, rerank="lexical", max_per_doc=1)), ["bus", "haksa", "water"])
            with self.assertRaises(ValueError):
                search(3, max_per_doc=0)


if __name__ == "__main__":
    unittest.main()
