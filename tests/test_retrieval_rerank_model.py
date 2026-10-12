"""리랭커 모델 위치: A는 공용 경로 로컬 폴더(다운로드 없음), B는 기존 HF ID. torch·transformers 없이 검증합니다."""

import contextlib
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import retrieval
from retrieval import rerank_model_source, retrieval_options

ENV_KEYS = ("RFP_SCENARIO", "A_RERANK_MODEL")


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def view(self, *_):
        return self

    def float(self):
        return self

    def tolist(self):
        return list(self.values)


class FakeEncoded(dict):
    def to(self, device):
        return self


class FakeModel:
    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, size):
        return SimpleNamespace(logits=FakeTensor([0.5] * size))


def fake_libraries(loads):
    """from_pretrained 호출 인자를 기록하는 가짜 torch·transformers 모듈."""
    def loader(result):
        def from_pretrained(name, **kwargs):
            loads.append((name, kwargs))
            return result
        return SimpleNamespace(from_pretrained=from_pretrained)
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), float16="f16", float32="f32",
                            no_grad=contextlib.nullcontext)
    tokenizer = lambda questions, passages, **kwargs: FakeEncoded(size=len(passages))
    transformers = SimpleNamespace(AutoModelForSequenceClassification=loader(FakeModel()),
                                   AutoTokenizer=loader(tokenizer))
    return {"torch": torch, "transformers": transformers}


class RerankModelSourceTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if not k.startswith("RETRIEVAL_") and k not in ENV_KEYS}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        available = patch.object(retrieval, "_cross_encoder_available", return_value=True)
        available.start()
        self.addCleanup(available.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.models = Path(directory.name)
        model_dir = patch.object(retrieval, "A_RERANK_MODEL_DIR", self.models)
        model_dir.start()
        self.addCleanup(model_dir.stop)

    def scenario_a(self, name=None):
        os.environ["RFP_SCENARIO"] = "A"
        if name is not None:
            os.environ["A_RERANK_MODEL"] = name

    def test_b_keeps_hf_id_and_override(self):
        self.assertEqual(rerank_model_source(), "BAAI/bge-reranker-v2-m3")
        os.environ["RETRIEVAL_RERANK_MODEL"] = "other/reranker"
        self.assertEqual(rerank_model_source(), "other/reranker")

    def test_b_ignores_a_rerank_model(self):
        os.environ["A_RERANK_MODEL"] = "missing"
        self.assertEqual(rerank_model_source(), "BAAI/bge-reranker-v2-m3")
        self.assertEqual(retrieval_options()["rerank"], "cross-encoder")

    def test_a_uses_shared_folder_not_hf_id(self):
        (self.models / "bge-reranker").mkdir()
        self.scenario_a(" bge-reranker ")
        os.environ["RETRIEVAL_RERANK_MODEL"] = "other/reranker"  # A에서는 무시
        self.assertEqual(rerank_model_source(), str(self.models / "bge-reranker"))
        self.assertEqual(retrieval_options()["rerank"], "cross-encoder")

    def test_a_default_without_model_falls_back_to_lexical(self):
        self.scenario_a()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(retrieval_options()["rerank"], "lexical")
        self.assertTrue(any("A_RERANK_MODEL" in str(item.message) for item in caught))

    def test_a_explicit_cross_encoder_without_model_fails(self):
        self.scenario_a()
        with self.assertRaisesRegex(ValueError, "A_RERANK_MODEL"):
            retrieval_options(rerank="cross-encoder")

    def test_a_rejects_path_like_names(self):
        for name in (".", "..", "a/b", "a\\b"):
            with self.subTest(name=name):
                self.scenario_a(name)
                with self.assertRaisesRegex(ValueError, "단일 모델 하위 폴더"):
                    retrieval_options(rerank="cross-encoder")

    def test_a_missing_folder_fails_before_search(self):
        self.scenario_a("missing")
        with self.assertRaisesRegex(ValueError, "리랭커 모델 폴더가 없습니다"):
            retrieval_options()

    def test_a_lexical_needs_no_model(self):
        self.scenario_a("missing")
        os.environ["RETRIEVAL_RERANK"] = "lexical"
        self.assertEqual(retrieval_options()["rerank"], "lexical")

    def test_local_folder_loads_without_hub(self):
        (self.models / "bge-reranker").mkdir()
        loads = []
        with patch.dict(sys.modules, fake_libraries(loads)), patch.dict(retrieval._cross_encoders, clear=True):
            scores = retrieval._cross_encoder_scores("질문", ["가", "나"], str(self.models / "bge-reranker"))
        self.assertEqual(scores, [0.5, 0.5])
        self.assertEqual(len(loads), 2)
        self.assertTrue(all(kwargs["local_files_only"] is True for _, kwargs in loads))

    def test_hf_id_keeps_hub_loading(self):
        loads = []
        with patch.dict(sys.modules, fake_libraries(loads)), patch.dict(retrieval._cross_encoders, clear=True):
            retrieval._cross_encoder_scores("질문", ["가"], "BAAI/bge-reranker-v2-m3")
        self.assertTrue(all(kwargs["local_files_only"] is False for _, kwargs in loads))


if __name__ == "__main__":
    unittest.main()
