"""외부 통신 없이 선택적 추적, 오류 격리, 본문 제외를 검증합니다."""

from contextlib import contextmanager
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiment_reports import save_report
from observability import Trace, model_call, observed


class FakeObservation:
    trace_id = "trace-test"

    def __init__(self, record):
        self.record = record

    def update(self, **kwargs):
        self.record.update(kwargs)


class FakeLangfuse:
    def __init__(self):
        self.records = []
        self.scores = []
        self.flushed = False

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.records.append(kwargs)
        yield FakeObservation(kwargs)

    def create_score(self, **kwargs):
        self.scores.append(kwargs)

    def flush(self):
        self.flushed = True


class ObservabilityTests(unittest.TestCase):
    def test_disabled_and_invalid_config(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(Trace.from_env({}).client)
        with patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test",
                                    "LANGFUSE_BASE_URL": "[https://us.cloud.langfuse.com](https://us.cloud.langfuse.com)"}, clear=True):
            with self.assertWarns(UserWarning):
                self.assertIsNone(Trace.from_env({}).client)

    def test_tokens_scores_and_flush_without_content(self):
        client = FakeLangfuse()
        trace = Trace({"owner": "tester"}, client)

        @observed("stage")
        def execute(secret_text):
            return model_call("generation", "test-model", lambda: SimpleNamespace(
                output_text=secret_text,
                usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14)))

        with trace.run("evaluate"):
            result = execute("PRIVATE_DOCUMENT")
            trace.scores({"recall_at_k": 1, "keyword_coverage": None})
        self.assertEqual(result.output_text, "PRIVATE_DOCUMENT")
        self.assertEqual(trace.usage["test-model"]["total"], 14)
        self.assertNotIn("PRIVATE_DOCUMENT", str(client.records))
        self.assertEqual(client.scores[0]["trace_id"], "trace-test")
        self.assertTrue(client.flushed)

    def test_pipeline_error_propagates_with_sanitized_error(self):
        client = FakeLangfuse()
        trace = Trace({}, client)
        with self.assertRaisesRegex(ValueError, "PRIVATE_ERROR"):
            with trace.run("test"):
                raise ValueError("PRIVATE_ERROR")
        self.assertNotIn("PRIVATE_ERROR", str(client.records))
        self.assertEqual(client.records[0]["status_message"], "ValueError")
        self.assertTrue(client.flushed)

    def test_exporter_failure_does_not_repeat_pipeline(self):
        client = FakeLangfuse()
        client.start_as_current_observation = lambda **kwargs: (_ for _ in ()).throw(RuntimeError())
        trace = Trace({}, client)
        calls = []
        with self.assertWarns(UserWarning):
            with trace.run("test"):
                calls.append(1)
        self.assertEqual(calls, [1])

    def test_share_report_excludes_filter_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_report(directory, "20261001", {"count": 1, "recall_at_k": 1},
                               {"filters": {"agency": "PRIVATE_DOCUMENT"}, "top_k": 5},
                               {"owner": "tester", "evaluation_sha256": "hash"}, {})
            text = Path(path).read_text()
            self.assertNotIn("PRIVATE_DOCUMENT", text)
            self.assertIn("evaluation_sha256", text)
            self.assertIn("recall_at_k", text)

    def test_evaluate_cli_writes_local_result_and_share_report(self):
        from embedding import build_index
        from test_pipeline import FakeClient
        import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeClient()
            build_index([{"doc_id": "a", "text": "예산 PRIVATE_DOCUMENT", "metadata": {
                "filename": "private.hwp"}}], client, root / "index")
            eval_file = root / "eval.json"
            eval_file.write_text(json.dumps([{"question": "예산 PRIVATE_QUESTION",
                                              "expected_doc_ids": ["a"],
                                              "expected_keywords": ["100원"]}]))
            argv = ["run.py", "evaluate", "--eval-file", str(eval_file),
                    "--index-dir", str(root / "index"), "--results-dir", str(root / "results"),
                    "--reports-dir", str(root / "reports"), "--owner", "tester"]
            with patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "LANGFUSE_ENABLED": "false"}, clear=True), \
                    patch("sys.argv", argv), patch("dotenv.load_dotenv"), \
                    patch("openai.OpenAI", return_value=client), patch.object(run, "ROOT", root):
                run.main()
            result = json.loads(next((root / "results").glob("evaluate_*.json")).read_text())
            self.assertEqual(result["summary"]["recall_at_k"], 1)
            self.assertEqual(result["experiment"]["experiment_id"], "tester-0001")
            self.assertIsNotNone(result["experiment"]["index_sha256"])
            self.assertIsNone(result["experiment"]["trace_id"])
            report = next((root / "reports").glob("*.md")).read_text()
            self.assertNotIn("PRIVATE_QUESTION", report)
            self.assertNotIn("PRIVATE_DOCUMENT", report)
            self.assertNotIn("private.hwp", report)


if __name__ == "__main__":
    unittest.main()
