"""외부 API·.env 접근 없이 모듈 계약과 검색·평가 동작을 검증합니다."""

import tempfile
from types import SimpleNamespace
import unittest

from embedding import build_index, chunk_documents, load_index
from evaluation import evaluate
from generation import generate_answer
from retrieval import retrieve
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정


class FakeClient:
    def __init__(self):
        self.embeddings = SimpleNamespace(create=self.embed)
        self.responses = SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text="예산은 100원입니다. [1]"))
        self.models = []

    def embed(self, model, input):
        self.models.append(model)
        # 역순 API 응답에서도 입력-벡터 대응이 유지되어야 합니다.
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1., 0.] if "예산" in text else [0., 1.])
                                     for i, text in reversed(list(enumerate(input)))])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            {"doc_id": "a", "text": "예산 100원", "metadata": {"발주 기관": "가", "filename": "a.hwp"}},
            {"doc_id": "b", "text": "일정 10월", "metadata": {"발주 기관": "나", "filename": "b.hwp"}},
        ]

    def test_index_search_generation_evaluation(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            build_index(self.documents, client, directory, model="test-embedding")
            index, chunks, config = load_index(directory)
            hits = retrieve("예산", client, index, chunks, config, top_k=1)
            self.assertEqual(hits[0]["doc_id"], "a")
            self.assertAlmostEqual(hits[0]["score"], 1.)
            filtered = retrieve("예산", client, index, chunks, config, top_k=1, filters={"발주 기관": "나"})
            self.assertEqual(filtered[0]["doc_id"], "b")
            self.assertEqual(retrieve("예산", client, index, chunks, config, filters={"발주 기관": "없음"}), [])
            result = generate_answer("예산", hits, client)
            report = evaluate([{"question": "예산", "expected_doc_ids": ["a", "b"], "expected_keywords": ["100원"]}],
                              lambda question, filters: result)
            self.assertEqual(report["summary"]["recall_at_k"], .5)
            self.assertEqual(report["summary"]["keyword_coverage"], 1.)
            self.assertEqual(set(client.models), {"test-embedding"})

    def test_chunk_coverage_and_offsets(self):
        doc = {"doc_id": "c", "text": "abcdefghijk", "metadata": {}}
        chunks = chunk_documents([doc], chunk_size=5, chunk_overlap=2)
        self.assertEqual([c["text"] for c in chunks], ["abcde", "defgh", "ghijk"])
        for chunk in chunks:
            self.assertEqual(chunk["text"], doc["text"][chunk["metadata"]["start_char"]:chunk["metadata"]["end_char"]])
        with self.assertRaises(ValueError):
            chunk_documents([doc], 5, 5)

    def test_no_hits_needs_no_api(self):
        result = generate_answer("예산", [], None)
        self.assertEqual(result["sources"], [])
        self.assertIn("없어", result["answer"])

    def test_evaluation_requires_manual_labels(self):
        with self.assertRaises(ValueError):
            evaluate([{"question": "예산", "expected_doc_ids": []}], None)


if __name__ == "__main__":
    unittest.main()
