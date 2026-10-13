"""리랭커 모델 위치: A는 서버 공용 폴더, B는 기존 HF 모델 ID. torch·transformers 없이 검증합니다."""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import retrieval
from retrieval import rerank_model_source, retrieval_options


class RerankModelSourceTests(unittest.TestCase):
    def setUp(self):
        # 개발자 PC의 RETRIEVAL_*·A 설정이 결과에 섞이지 않게 비웁니다.
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                      if not k.startswith("RETRIEVAL_") and k not in ("RFP_SCENARIO", "A_RERANK_MODEL")},
                         clear=True)
        env.start()
        self.addCleanup(env.stop)
        # torch·transformers가 설치된 것처럼 두어야 기본 리랭킹이 cross-encoder가 됩니다.
        available = patch.object(retrieval, "_cross_encoder_available", return_value=True)
        available.start()
        self.addCleanup(available.stop)
        # 서버 공용 폴더 대신 임시 폴더를 씁니다.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.models = Path(directory.name)
        model_dir = patch.object(retrieval, "A_RERANK_MODEL_DIR", self.models)
        model_dir.start()
        self.addCleanup(model_dir.stop)

    def test_b_uses_hf_id_and_ignores_a_setting(self):
        os.environ["A_RERANK_MODEL"] = "missing"
        self.assertEqual(rerank_model_source(), "BAAI/bge-reranker-v2-m3")
        os.environ["RETRIEVAL_RERANK_MODEL"] = "other/reranker"
        self.assertEqual(rerank_model_source(), "other/reranker")
        self.assertEqual(retrieval_options()["rerank"], "cross-encoder")

    def test_a_uses_shared_folder(self):
        (self.models / "bge-reranker").mkdir()
        os.environ.update(RFP_SCENARIO="A", A_RERANK_MODEL="bge-reranker",
                          RETRIEVAL_RERANK_MODEL="other/reranker")  # A에서는 무시
        self.assertEqual(rerank_model_source(), str(self.models / "bge-reranker"))
        self.assertEqual(retrieval_options()["rerank"], "cross-encoder")

    def test_a_without_model_defaults_to_lexical(self):
        os.environ["RFP_SCENARIO"] = "A"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(retrieval_options()["rerank"], "lexical")
        self.assertTrue(any("A_RERANK_MODEL" in str(item.message) for item in caught))

    def test_a_missing_folder_fails(self):
        os.environ["RFP_SCENARIO"] = "A"
        for name in ("", "missing"):
            with self.subTest(name=name):
                os.environ["A_RERANK_MODEL"] = name
                with self.assertRaisesRegex(ValueError, "리랭커 모델 폴더가 없습니다"):
                    rerank_model_source()


if __name__ == "__main__":
    unittest.main()
