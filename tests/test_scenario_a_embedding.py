"""Local embedding contract and privacy tests; no model downloads or GPU required."""
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from a_embedding import build_index, embed_texts, load_index
from a_evaluation import evaluate
from a_generation import generate_answer
from a_observability import Trace
from a_retrieval import retrieve
from a_client import ScenarioAClient
from a_local_embedding import _LocalEmbedding, load_model
from test_observability import FakeLangfuse


class FakeOOM(RuntimeError):
    pass


def fake_torch(cuda=False):
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda,
                           OutOfMemoryError=FakeOOM, empty_cache=Mock()),
                           inference_mode=nullcontext, float16="fp16", float32="fp32")


class FakeModel:
    max_seq_length = 5

    def __init__(self, capacity=100):
        self.capacity, self.batches = capacity, []

    def __getitem__(self, index):
        return SimpleNamespace(do_lower_case=False)

    def tokenizer(self, texts, **kwargs):
        return {"input_ids": [[0] + list(text) + [1] for text in texts]}

    def eval(self):
        return self

    def encode(self, texts, **kwargs):
        self.batches.append(list(texts))
        if len(texts) > self.capacity:
            raise FakeOOM("PRIVATE_OOM")
        return np.array([[len(text) + 1, sum(map(ord, text)) % 17 + 1] for text in texts], dtype=np.float16)


class LocalEmbeddingTests(unittest.TestCase):
    def test_import_does_not_load_optional_packages_or_env(self):
        code = ("import sys; import a_local_embedding as scenario_a_embedding; "
                "assert 'torch' not in sys.modules; "
                "assert 'sentence_transformers' not in sys.modules; "
                "assert 'dotenv' not in sys.modules")
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_loader_is_local_and_chooses_device_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "modules.json").write_text("[]")
            for cuda in (False, True):
                model = FakeModel()
                model.max_seq_length = 10000
                factory = Mock(return_value=model)
                with patch.dict(sys.modules, {"torch": fake_torch(cuda),
                        "sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
                    obj = load_model(path)
                    obj.embed_texts(["a"])
                    obj.embed_texts(["b"])
                factory.assert_called_once_with(str(path.resolve()), device="cuda" if cuda else "cpu",
                    local_files_only=True, trust_remote_code=False,
                    model_kwargs={"dtype": "fp16" if cuda else "fp32"})
                self.assertEqual(model.max_seq_length, 8192)

    def test_missing_or_unprepared_model_rejected_before_dependency_import(self):
        with tempfile.TemporaryDirectory() as directory:
            for path in (Path(directory), Path(directory) / "missing"):
                with self.assertRaises(OSError):
                    load_model(path)

    def test_order_dtype_empty_string_and_batch_boundaries(self):
        model = FakeModel()
        obj = _LocalEmbedding(model, fake_torch(), "model")
        texts = ["a" * i for i in range(19)]
        actual = obj.embed_texts(texts)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (19, 2))
        np.testing.assert_array_equal(actual[:, 0], np.arange(1, 20))
        self.assertEqual(list(map(len, model.batches)), [8, 8, 3])

    def test_invalid_inputs_never_infer(self):
        model = FakeModel()
        obj = _LocalEmbedding(model, fake_torch(), "model")
        for texts in ([], (), "text", None, [None], [1], ["ok", {}]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                obj.embed_texts(texts)
        self.assertEqual(model.batches, [])

    def test_token_counts_truncation_and_trace_privacy(self):
        model = FakeModel()
        obj = _LocalEmbedding(model, fake_torch(), "model")
        fake = FakeLangfuse()
        trace = Trace({}, fake)
        with trace.run("test"):
            obj.embed_texts(["", " a ", "PRIVATE_TEXT"])
        self.assertEqual(trace.usage["local/model"], {"input": 10, "output": 0, "total": 10})
        self.assertEqual(obj.stats["truncated_texts"], 1)
        self.assertNotIn("PRIVATE_TEXT", str(fake.records))
        self.assertTrue(fake.flushed)

    def test_oom_retries_keep_order_and_do_not_double_count_tokens(self):
        model, torch = FakeModel(capacity=2), fake_torch(True)
        obj = _LocalEmbedding(model, torch, "model")
        result = obj.embed_texts(["a" * i for i in range(8)])
        np.testing.assert_array_equal(result[:, 0], np.arange(1, 9))
        self.assertEqual(obj.batch_size, 2)
        self.assertEqual(obj.stats["oom_retries"], 2)
        self.assertEqual(obj.stats["input_tokens"], sum(min(i + 2, 5) for i in range(8)))
        self.assertEqual(torch.cuda.empty_cache.call_count, 2)

    def test_single_item_oom_stops_and_trace_sanitizes_error(self):
        obj = _LocalEmbedding(FakeModel(capacity=0), fake_torch(True), "model")
        fake, trace = FakeLangfuse(), None
        trace = Trace({}, fake)
        with self.assertRaises(FakeOOM), trace.run("test"):
            obj.embed_texts(["secret"])
        self.assertEqual(obj.stats["failures"], 1)
        self.assertEqual(trace.usage, {})
        self.assertNotIn("PRIVATE_OOM", str(fake.records))
        self.assertNotIn("secret", str(fake.records))

    def test_trace_export_failure_does_not_repeat_inference(self):
        model = FakeModel()
        obj = _LocalEmbedding(model, fake_torch(), "model")
        fake = FakeLangfuse()
        fake.start_as_current_observation = Mock(side_effect=RuntimeError)
        with self.assertWarns(UserWarning), Trace({}, fake).run("test"):
            obj.embed_texts(["private"])
        self.assertEqual(len(model.batches), 1)

    def test_invalid_local_vectors_are_rejected_without_overwriting_index(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                {"RFP_CHUNKING_STRATEGY": "fixed", "RFP_EMBEDDING_CONTEXT": "none"}):
            path = Path(directory)
            model = FakeModel()
            client = ScenarioAClient.__new__(ScenarioAClient)
            client.embedding_model = "emb"
            client.embedding = _LocalEmbedding(model, fake_torch(), "emb")
            docs = [{"doc_id": "d", "text": "body", "metadata": {}}]
            build_index(docs, client, path, "emb")
            before = {p.name: p.read_bytes() for p in path.iterdir()}
            for invalid in ([[0., 0.]], [[float("nan"), 1.]], [[1., float("inf")]], [[]], [[1., 2.], [3., 4.]]):
                with self.subTest(invalid=invalid), patch.object(model, "encode", return_value=invalid):
                    with self.assertRaises(ValueError):
                        build_index(docs, client, path, "emb")
                self.assertEqual(before, {p.name: p.read_bytes() for p in path.iterdir()})

    def test_actual_adapter_through_index_search_answer_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,
                {"RFP_CHUNKING_STRATEGY": "fixed", "RFP_EMBEDDING_CONTEXT": "none"}):
            root = Path(directory)
            (root / "emb").mkdir()
            (root / "emb/modules.json").write_text("[]")
            model = FakeModel()
            factory = Mock(return_value=model)
            with patch.dict(sys.modules, {"torch": fake_torch(),
                    "sentence_transformers": SimpleNamespace(SentenceTransformer=factory)}):
                client = ScenarioAClient({"A_EMBEDDING_MODEL": "emb", "A_GENERATION_MODEL": "gen"},
                                         root, root, need_generation=False)
            self.assertIsNone(client.generation)
            documents = [{"doc_id": "d", "text": "PRIVATE_DOCUMENT", "metadata": {"filename": "private"}}]
            config = build_index(documents, client, root / "index", "emb", documents_sha256="hash")
            index, chunks, loaded = load_index(root / "index")
            self.assertEqual(loaded, config)
            self.assertEqual(chunks[0]["text"], documents[0]["text"])
            self.assertAlmostEqual(float(np.linalg.norm(index.reconstruct(0))), 1, places=6)
            client.generation = SimpleNamespace(generate_answer=lambda q, h, m: {
                "question": q, "answer": "answer", "model": m, "status": "ok",
                "sources": [{"citation": i, **hit} for i, hit in enumerate(h, 1)]})
            def answer(question, filters):
                hits = retrieve(question, client, index, chunks, loaded, 5, filters,
                                rerank="none", hybrid=False, rewrite="off", within_doc="off", max_per_doc="none",
                                min_score="none")  # 가짜 임베딩은 코사인이 낮아 범위 밖 차단을 끕니다
                return generate_answer(question, hits, client, "gen")
            result = evaluate([{"question": "q", "expected_doc_ids": ["d"]}], answer)
            self.assertEqual(result["summary"]["recall_at_k"], 1)
            self.assertEqual(result["records"][0]["sources"][0]["citation"], 1)
            json.dumps(result)
            with self.assertRaisesRegex(ValueError, "모델"):
                embed_texts(["q"], client, "different-model")
            factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
