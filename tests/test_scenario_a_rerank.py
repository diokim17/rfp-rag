"""A 파이프라인(build→ask)에서 검색 기본값과 공용 경로 리랭커가 실제로 연결되는지 오프라인으로 검증합니다.

A 임베딩·생성 모델과 torch·transformers는 모두 가짜이며 OpenAI·Hub 접속이 없습니다.
"""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import run
import retrieval
from observability import Trace
from test_retrieval_rerank_model import FakeModel, FakeTensor, fake_libraries

QUESTION = "예산은 얼마인가요?"
# 코사인은 first가 높고(질문과 같은 벡터), 리랭커만 second를 위로 올립니다.
DOCS = [{"doc_id": "first", "text": "일반 안내 문서", "metadata": {"filename": "first.hwp", "사업명": "가 사업"}},
        {"doc_id": "second", "text": "정답 근거가 있는 문서", "metadata": {"filename": "second.hwp", "사업명": "나 사업"}}]


class PassageModel(FakeModel):
    """'정답'이 든 후보에 높은 점수를 주는 가짜 cross-encoder."""

    def __call__(self, passages):
        return SimpleNamespace(logits=FakeTensor([10.0 if "정답" in passage else 0.0 for passage in passages]))


def rerank_libraries(loads):
    libraries = fake_libraries(loads)
    transformers = libraries["transformers"]

    def load(name, **kwargs):
        loads.append((name, kwargs))
        return PassageModel()

    tokenizer = lambda questions, passages, **kwargs: SimpleNamespace(to=lambda device: {"passages": passages})
    transformers.AutoModelForSequenceClassification = SimpleNamespace(from_pretrained=load)
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=lambda name, **kwargs: tokenizer)
    return libraries


class ScenarioARerankTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = root = Path(directory.name)
        self.gen, self.emb, self.rerank = root / "generate", root / "embedding", root / "rerank"
        for path in (self.gen / "gen", self.emb / "emb", self.rerank / "reranker"):
            path.mkdir(parents=True)
        embedding = SimpleNamespace(embed_texts=Mock(side_effect=lambda texts: [
            [0.6, 0.8] if "정답" in text else [1., 0.] for text in texts]))

        def answer(question, hits, model):
            return {"question": question, "answer": "답변", "model": model,
                    "sources": [{"citation": i, **hit} for i, hit in enumerate(hits, 1)]}
        self.modules = {
            "scenario_a_embedding": SimpleNamespace(load_model=Mock(return_value=embedding)),
            "scenario_a_generation": SimpleNamespace(load_model=Mock(
                return_value=SimpleNamespace(generate_answer=Mock(side_effect=answer)))),
        }
        self.args = SimpleNamespace(
            command="all", raw_dir=root / "raw", processed_dir=root / "processed", index_dir=root / "index",
            results_dir=root / "results", reports_dir=root / "reports", limit=2, owner="test", chunk_size=1000,
            chunk_overlap=150, top_k=1, question=QUESTION, eval_file=None,
            scenario_settings={"scenario": "A", "A_GENERATION_MODEL": "gen", "A_EMBEDDING_MODEL": "emb"})

    def run_a(self, env, loads):
        """RFP_SCENARIO=A와 env만 둔 채(RETRIEVAL_* 기본값) all을 실행하고 결과 JSON을 돌려줍니다."""
        def parse(raw, processed, limit):
            run.write_json(processed / "documents.json", DOCS)
            run.write_json(processed / "parsing_errors.json", [])
            return DOCS
        with patch.dict(os.environ, {"RFP_SCENARIO": "A", **env}, clear=True), \
                patch.dict("sys.modules", {**self.modules, **rerank_libraries(loads)}), \
                patch.dict(retrieval._cross_encoders, clear=True), \
                patch.object(retrieval, "_cross_encoder_available", return_value=True), \
                patch.object(retrieval, "A_RERANK_MODEL_DIR", self.rerank), \
                patch.object(run, "A_GENERATION_MODEL_DIR", self.gen), \
                patch.object(run, "A_EMBEDDING_MODEL_DIR", self.emb), \
                patch("run.parse_documents", side_effect=parse), \
                patch("run.OpenAI") as openai, redirect_stdout(io.StringIO()):
            run.run_pipeline(self.args, run.argparse.ArgumentParser(), {}, {"experiment_id": "test-0001"},
                             Trace({}), "test")
        openai.assert_not_called()
        return run.read_json(self.args.results_dir / "all_test-0001_test.json")

    def test_default_cross_encoder_loads_shared_folder_and_reorders(self):
        loads = []
        result = self.run_a({"A_RERANK_MODEL": "reranker"}, loads)
        self.assertEqual([source["doc_id"] for source in result["sources"]], ["second"])
        self.assertIn("rerank_score", result["sources"][0])
        self.assertEqual(loads, [(str(self.rerank / "reranker"), {"dtype": "f32", "local_files_only": True})])

    def test_without_rerank_cosine_keeps_first(self):
        """대조군: 리랭킹이 없으면 first가 1위라 위 테스트의 순서 변화는 리랭커 때문입니다."""
        result = self.run_a({"RETRIEVAL_RERANK": "none"}, [])
        self.assertEqual([source["doc_id"] for source in result["sources"]], ["first"])

    def test_without_rerank_model_falls_back_to_lexical_without_loading(self):
        loads = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self.run_a({}, loads)
        self.assertEqual(loads, [])
        self.assertEqual(len(result["sources"]), 1)
        self.assertTrue(any("A_RERANK_MODEL" in str(item.message) for item in caught))

    def test_missing_rerank_folder_stops_before_generation(self):
        with self.assertRaisesRegex(ValueError, "리랭커 모델 폴더가 없습니다"):
            self.run_a({"A_RERANK_MODEL": "missing"}, [])
        generation = self.modules["scenario_a_generation"].load_model.return_value
        generation.generate_answer.assert_not_called()

    def test_a_rejects_rewrite_in_retrieve(self):
        """run.py를 거치지 않는 실험 스크립트에서도 A 재작성은 원문 폴백 대신 오류로 막습니다."""
        for mode in ("only", "both"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"RFP_SCENARIO": "A"}, clear=True), \
                    self.assertRaisesRegex(ValueError, "RETRIEVAL_REWRITE=off"):
                retrieval.retrieval_options(rerank="lexical", rewrite=mode)

    def test_b_still_allows_rewrite_option(self):
        with patch.dict(os.environ, {"RFP_SCENARIO": "B"}, clear=True):
            self.assertEqual(retrieval.retrieval_options(rerank="lexical", rewrite="only")["rewrite"], "only")


if __name__ == "__main__":
    unittest.main()
