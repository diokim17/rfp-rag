"""Validate local evaluation coverage, metrics, failures and report privacy."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.a_embedding_chy import measured_cases, summarize, validate_inputs, report, verify_model, sha256
from parsing.parsing_documents import write_json


class ExperimentTests(unittest.TestCase):
    def test_missing_gold_and_duplicate_documents_are_rejected(self):
        cases = [{"id": "q1", "question": "private", "expected_doc_ids": ["a"]}]
        validate_inputs([{"doc_id": "a"}], cases)
        for docs in ([], [{"doc_id": "b"}], [{"doc_id": "a"}, {"doc_id": "a"}]):
            with self.assertRaises(ValueError):
                validate_inputs(docs, cases)

    def test_repeat_metrics_errors_and_search_options(self):
        cases = [{"id": "q1", "question": "PRIVATE_QUESTION", "expected_doc_ids": ["a", "b"]}]
        hits = [{"doc_id": "x", "chunk_id": "x:0", "score": 1.0},
                {"doc_id": "a", "chunk_id": "a:0", "score": .5}]
        with patch("experiments.a_embedding_chy.retrieve", side_effect=[hits, ValueError("PRIVATE_ERROR")]) as search:
            rows = measured_cases(cases, None, None, None, None, 2)
        self.assertEqual(rows[0]["recall"], .5)
        self.assertEqual(rows[0]["mrr"], .5)
        self.assertEqual(rows[1]["error"], "ValueError")
        self.assertEqual(summarize(rows)["failures"], 1)
        self.assertFalse(search.call_args.kwargs["hybrid"])
        self.assertEqual(search.call_args.kwargs["rerank"], "none")
        self.assertNotIn("PRIVATE_", str(rows))

    def test_report_excludes_nonallowlisted_data(self):
        result = {key: {} for key in ("experiment", "model", "settings", "summary", "timing", "stats")}
        result.update(question="PRIVATE_QUESTION", records=[{
            "id": "q1", "repetition": 0, "recall": 0, "mrr": 0, "error": None,
            "question": "PRIVATE_QUESTION", "hits": [{"text": "PRIVATE_CHUNK"}]}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            report(result, path)
            content = path.read_text()
        self.assertNotIn("PRIVATE_", content)
        self.assertIn("q1", content)

    def test_model_file_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            weights = path / "model.safetensors"
            weights.write_bytes(b"original")
            manifest = {"files": {weights.name: sha256(weights)}}
            write_json(path / "local_manifest.json", manifest)
            self.assertEqual(verify_model(path), manifest)
            weights.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "manifest"):
                verify_model(path)


if __name__ == "__main__":
    unittest.main()
