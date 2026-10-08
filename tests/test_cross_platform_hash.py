"""운영체제(Windows CRLF, macOS·Linux LF)와 무관하게 같은 문서·인덱스로 인정되는지 검증합니다. API는 쓰지 않습니다."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run
from experiment_reports import file_hash, text_file_hash, text_hash
from observability import Trace
from parsing import read_json, write_json

DOCUMENTS = [{"doc_id": "a", "text": "예산\n줄바꿈이 든 본문\r\n", "metadata": {"filename": "a.hwp", "source": "files/a.hwp"}}]


def crlf(path):
    """Windows에서 Path.write_text로 저장한 것처럼 줄바꿈을 CRLF로 바꿉니다."""
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))


class TextHashTests(unittest.TestCase):
    def test_write_json_always_writes_lf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.json"
            write_json(path, DOCUMENTS)
            data = path.read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertEqual(read_json(path), DOCUMENTS)  # 본문 안의 \r\n은 이스케이프되어 그대로 보존

    def test_crlf_and_lf_files_share_text_hash_but_not_byte_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            lf, windows = Path(directory) / "lf.json", Path(directory) / "crlf.json"
            write_json(lf, DOCUMENTS)
            write_json(windows, DOCUMENTS)
            crlf(windows)
            self.assertEqual(text_file_hash(lf), text_file_hash(windows))
            self.assertEqual(text_file_hash(lf), file_hash(lf))  # LF 파일은 예전 바이트 해시와 같음
            self.assertNotEqual(file_hash(lf), file_hash(windows))
            self.assertEqual(text_hash(b"a\r\nb\rc\n"), hashlib.sha256(b"a\nb\nc\n").hexdigest())
            self.assertIsNone(text_file_hash(Path(directory) / "missing.json"))

    def test_different_content_still_differs(self):
        self.assertNotEqual(text_hash(b'{"a": 1}\n'), text_hash(b'{"a": 2}\n'))


class IndexAcrossPlatformsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.args = SimpleNamespace(
            command="ask", processed_dir=root / "processed", index_dir=root / "index", owner="tester",
            raw_dir=root / "raw", limit=None, chunk_size=1000, chunk_overlap=150, top_k=5,
            question="예산?", results_dir=root / "results")
        self.documents_path = self.args.processed_dir / "documents.json"
        write_json(self.documents_path, DOCUMENTS)
        self.lf_hash = file_hash(self.documents_path)
        crlf(self.documents_path)
        self.crlf_hash = file_hash(self.documents_path)

    def check(self, stored):
        write_json(self.args.index_dir / "config.json", {"documents_sha256": stored, "chunk_count": 1})
        run.check_index_documents(self.args, argparse.ArgumentParser())

    def test_index_built_on_linux_or_mac_accepts_windows_documents(self):
        self.check(self.lf_hash)  # SystemExit이 나지 않으면 통과

    def test_index_built_on_windows_before_fix_is_still_accepted(self):
        self.check(self.crlf_hash)

    def test_changed_documents_are_still_rejected(self):
        write_json(self.documents_path, [{**DOCUMENTS[0], "text": "다른 본문"}])
        with redirect_stderr(io.StringIO()) as output, self.assertRaises(SystemExit):
            self.check(self.lf_hash)
        self.assertIn("생성 당시와 다릅니다", output.getvalue())

    def test_build_stores_lf_hash_for_crlf_documents(self):
        self.args.command = "build"
        stored = {}

        def build(documents, client, index_dir, model, size, overlap, *, documents_sha256):
            self.assertEqual(documents, DOCUMENTS)
            stored["hash"] = documents_sha256
            config = {"documents_sha256": documents_sha256, "chunk_count": 1}
            write_json(index_dir / "config.json", config)
            return config

        experiment = {"experiment_id": "tester-0001"}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "OPENAI_GENERATION_MODEL": "gpt-5-mini"}), \
                patch("run.OpenAI"), patch("run.build_index", side_effect=build), redirect_stdout(io.StringIO()):
            run.run_pipeline(self.args, argparse.ArgumentParser(), {}, experiment, Trace({}), "test")
        self.assertEqual(stored["hash"], self.lf_hash)
        self.assertEqual(experiment["documents_sha256"], self.lf_hash)


if __name__ == "__main__":
    unittest.main()
