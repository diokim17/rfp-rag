"""박단비: 시나리오 A(pipelines.server.server_run)에서도 B와 같은 OpenAI 판정기가 연결되는지 확인합니다 (모델·API는 mock)."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipelines.server import server_run
from observability.observability_trace import Trace
from parsing.parsing_documents import read_json, write_json


class AJudgeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.args = SimpleNamespace(
            command="evaluate", processed_dir=root / "processed", index_dir=root / "index", owner="tester",
            top_k=5, results_dir=root / "results", reports_dir=root / "reports", eval_file=root / "eval.json",
            scenario_settings={"scenario": "A", "SERVER_GENERATION_MODEL": "g", "SERVER_EMBEDDING_MODEL": "e"})
        write_json(self.args.processed_dir / "documents.json", [{"doc_id": "a", "text": "예산", "metadata": {}}])
        digest = hashlib.sha256((self.args.processed_dir / "documents.json").read_bytes()).hexdigest()
        self.config = {"documents_sha256": digest, "chunk_count": 1}
        write_json(self.args.index_dir / "config.json", self.config)
        write_json(self.args.eval_file, [{"id": "llm-1", "grading": "llm"}])

    def run_evaluate(self, judge_model):
        self.args.judge_model = judge_model
        self.args.judge_cache = None
        output = io.StringIO()
        with patch("pipelines.server.server_client.ScenarioAClient") as local_client, patch("pipelines.server.server_run.OpenAI") as openai_client, \
                patch("pipelines.server.server_run.load_index", return_value=(object(), [], self.config)), \
                patch("pipelines.server.server_run.make_llm_judge") as factory, \
                patch("pipelines.server.server_run.evaluate", return_value={"summary": {}}) as evaluate, \
                patch("pipelines.server.server_run.save_report"), redirect_stdout(io.StringIO()), redirect_stderr(output):
            server_run.run_pipeline(self.args, argparse.ArgumentParser(), {}, {"experiment_id": "tester-0001"},
                               Trace({}), "test")
        return local_client, openai_client, factory, evaluate, output.getvalue()

    def test_judge_uses_openai_client_not_local_model(self):
        local_client, openai_client, factory, evaluate, output = self.run_evaluate("gpt-5-nano")
        # 판정기에는 로컬 모델이 아니라 새로 만든 OpenAI 클라이언트가 들어가야 B와 같은 기준으로 채점됩니다.
        factory.assert_called_once_with(openai_client.return_value, model="gpt-5-nano", cache_path=None)
        self.assertIsNot(factory.call_args.args[0], local_client.return_value)
        self.assertIs(evaluate.call_args.kwargs["judge"], factory.return_value)
        self.assertNotIn("미채점", output)
        self.assertEqual(read_json(self.args.results_dir / "evaluate_tester-0001_test.json")["judge_settings"]["model"],
                         "gpt-5-nano")

    def test_without_judge_no_openai_call(self):
        _, openai_client, factory, evaluate, output = self.run_evaluate(None)
        openai_client.assert_not_called()
        factory.assert_not_called()
        self.assertIsNone(evaluate.call_args.kwargs["judge"])
        self.assertIn("LLM 1문항은 미채점", output)


if __name__ == "__main__":
    unittest.main()
