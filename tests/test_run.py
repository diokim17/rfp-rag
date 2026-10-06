"""인덱스 최신 여부와 빌드 해시 전달 검증. API와 실제 파싱은 mock으로 대체합니다."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run
from observability import Trace
from parsing import write_json


class IndexFreshnessTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.args = SimpleNamespace(
            command="ask", processed_dir=root / "custom processed",
            index_dir=root / "custom index", owner="tester", raw_dir=root / "raw",
            limit=3, chunk_size=1000, chunk_overlap=150, top_k=5,
            question="예산?", results_dir=root / "results")
        self.documents = [{"doc_id": "a", "text": "예산", "metadata": {"filename": "a.hwp"}}]
        write_json(self.args.processed_dir / "documents.json", self.documents)
        self.digest = hashlib.sha256((self.args.processed_dir / "documents.json").read_bytes()).hexdigest()
        self.config = {"documents_sha256": self.digest, "chunk_count": 1}
        write_json(self.args.index_dir / "config.json", self.config)
        self.parser = argparse.ArgumentParser()

    def pipeline(self):
        return run.run_pipeline(self.args, self.parser, {}, {"experiment_id": "tester-0001"}, Trace({}), "test")

    def assert_rejected(self, message):
        output = io.StringIO()
        with patch("run.OpenAI") as client, patch("run.retrieve") as retrieve, \
                redirect_stderr(output), self.assertRaises(SystemExit) as error:
            self.pipeline()
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, output.getvalue())
        client.assert_not_called()
        retrieve.assert_not_called()
        return output.getvalue()

    def test_matching_hash_continues_to_answer_with_fake_client(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "OPENAI_GENERATION_MODEL": "gpt-5-mini"}), \
                patch("run.OpenAI"), patch("run.load_index", return_value=(object(), [], self.config)), \
                patch("run.retrieve", return_value=[]) as retrieve, \
                patch("run.generate_answer", return_value={"answer": "근거 없음", "sources": []}), \
                redirect_stdout(io.StringIO()):
            self.pipeline()
        retrieve.assert_called_once()

    def test_ask_and_evaluate_reject_changed_documents_before_client_creation(self):
        write_json(self.args.processed_dir / "documents.json", [])
        for command in ("ask", "evaluate"):
            with self.subTest(command=command):
                self.args.command = command
                output = self.assert_rejected("생성 당시와 다릅니다")
                self.assertIn(str(self.args.processed_dir), output)
                self.assertIn(str(self.args.index_dir), output)
                self.assertIn("--owner tester", output)

    def test_legacy_and_invalid_hashes_are_rejected(self):
        for config in ({}, {"documents_sha256": None}, {"documents_sha256": "bad"}, []):
            with self.subTest(config=config):
                write_json(self.args.index_dir / "config.json", config)
                self.assert_rejected("유효한 documents_sha256이 없습니다")

    def test_missing_documents_are_rejected(self):
        (self.args.processed_dir / "documents.json").unlink()
        self.assert_rejected("전처리 문서가 없습니다")

    def test_missing_and_corrupt_config_are_rejected(self):
        path = self.args.index_dir / "config.json"
        path.unlink()
        self.assert_rejected("검증 파일을 읽을 수 없습니다")
        path.write_text("{", encoding="utf-8")
        self.assert_rejected("검증 파일을 읽을 수 없습니다")

    def test_build_and_all_pass_exact_input_hash_and_verify_saved_config(self):
        for command in ("build", "all"):
            with self.subTest(command=command):
                self.args.command = command
                write_json(self.args.processed_dir / "parsing_errors.json", [])

                def build(documents, client, index_dir, model, size, overlap, *, documents_sha256):
                    self.assertEqual(documents, self.documents)
                    self.assertEqual(documents_sha256, self.digest)
                    write_json(index_dir / "config.json", self.config)
                    return self.config

                with patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "OPENAI_GENERATION_MODEL": "gpt-5-mini"}), \
                        patch("run.OpenAI"), patch("run.parse_documents", return_value=self.documents), \
                        patch("run.build_index", side_effect=build) as builder, \
                        patch("run.load_index", return_value=(object(), [], self.config)), \
                        patch("run.retrieve", return_value=[]) as retrieve, \
                        patch("run.generate_answer", return_value={"answer": "근거 없음", "sources": []}), \
                        redirect_stdout(io.StringIO()):
                    self.pipeline()
                builder.assert_called_once()
                self.assertEqual(retrieve.call_count, int(command == "all"))

    def test_all_stops_before_retrieval_if_builder_does_not_save_hash(self):
        self.args.command = "all"
        write_json(self.args.processed_dir / "parsing_errors.json", [])
        write_json(self.args.index_dir / "config.json", {"chunk_count": 1})
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "OPENAI_GENERATION_MODEL": "gpt-5-mini"}), \
                patch("run.OpenAI"), patch("run.parse_documents", return_value=self.documents), \
                patch("run.build_index", return_value={"chunk_count": 1}), \
                patch("run.retrieve") as retrieve, redirect_stdout(io.StringIO()), \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.pipeline()
        retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
