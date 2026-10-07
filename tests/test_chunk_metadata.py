"""섹션·표 설명의 JSON 호환성과 원문 좌표, downstream 전달을 오프라인 검증."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from embedding import build_index, chunk_documents, load_index
from generation import generate_answer
from observability import Trace
from parsing import _number_tables, _render_table, _sections, read_json, section_of, write_json
from retrieval import retrieve
from test_observability import FakeLangfuse
from test_pipeline import FakeClient
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정


class SectionMetadataTests(unittest.TestCase):
    def test_uses_parser_helper_at_each_chunk_start_for_all_modes(self):
        text = "표지\nⅠ 사업 개요\n" + "개요 본문 " * 12 + "\nⅡ 요구사항\n" + "요구 본문 " * 12
        doc = {"doc_id": "a", "text": text, "metadata": {"사업명": "예시"}, "sections": _sections(text)}
        original = copy.deepcopy(doc)
        self.assertGreaterEqual(len(doc["sections"]), 2)
        for strategy in ("fixed", "boundary", "structured"):
            with self.subTest(strategy=strategy), patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}), \
                    patch("embedding.section_of", wraps=section_of) as helper:
                chunks = chunk_documents([doc], 60, 15)
                self.assertEqual(helper.call_count, len(chunks))
                for chunk, call in zip(chunks, helper.call_args_list):
                    start, end = (chunk["metadata"][k] for k in ("start_char", "end_char"))
                    self.assertEqual(call.args[1], start)
                    self.assertEqual(chunk["metadata"]["section_path"], section_of(doc, start))
                    self.assertEqual(chunk["text"], text[start:end])
                    self.assertEqual(chunk["metadata"]["사업명"], "예시")
                self.assertEqual(json.loads(json.dumps(chunks)), chunks)
        self.assertEqual(doc, original)

    def test_cross_section_chunk_is_labeled_by_start_including_preface(self):
        text = "표지\nⅠ 개요\n개요\nⅡ 요구\n내용"
        sections = [[3, "Ⅰ 개요"], [11, "Ⅱ 요구"]]
        doc = {"doc_id": "a", "text": text, "metadata": {}, "sections": sections}
        with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "fixed"}):
            self.assertEqual(chunk_documents([doc], len(text), 0)[0]["metadata"]["section_path"], "")
            chunks = chunk_documents([doc], 11, 0)
            self.assertEqual(chunks[1]["metadata"]["section_path"], "Ⅱ 요구")
        doc["text"] = text[3:]
        doc["sections"] = [[0, "Ⅰ 개요"], [8, "Ⅱ 요구"]]
        with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "structured"}):
            self.assertEqual(chunk_documents([doc], 100, 0)[0]["metadata"]["section_path"], "Ⅰ 개요")

    def test_missing_empty_and_malformed_sections_are_safe(self):
        text = "Ⅰ 개요\n본문 " * 5
        invalid = [None, [], {}, "invalid", [[True, "잘못됨"], [1, "줄 중간"], [-1, "음수"],
                                            [len(text), "끝 밖"], [0, None], [0], "invalid"]]
        for value in invalid:
            with self.subTest(value=value):
                doc = {"doc_id": "a", "text": text, "metadata": {}, "sections": value}
                original = copy.deepcopy(doc)
                self.assertTrue(all(c["metadata"]["section_path"] == "" for c in chunk_documents([doc], 10, 2)))
                self.assertEqual(doc, original)
        doc.pop("sections")
        self.assertTrue(all(c["metadata"]["section_path"] == "" for c in chunk_documents([doc], 10, 2)))

    def test_unsorted_sections_are_normalized_without_mutating_document(self):
        doc = {"doc_id": "a", "text": "Ⅰ 개요\n본문\nⅡ 요구\n내용", "metadata": {},
               "sections": [[8, "Ⅱ 요구"], [0, "Ⅰ 개요"]]}
        original = copy.deepcopy(doc)
        with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "fixed"}):
            chunks = chunk_documents([doc], 8, 0)
        self.assertEqual([c["metadata"]["section_path"] for c in chunks], ["Ⅰ 개요", "Ⅱ 요구"])
        self.assertEqual(doc, original)

    def test_section_metadata_roundtrip_retrieval_generation_and_trace_privacy(self):
        doc = {"doc_id": "a", "text": "Ⅰ 개요\n예산 100원", "metadata": {"filename": "a.hwp"},
               "sections": [[0, "PRIVATE_SECTION"]]}
        client = FakeClient()
        client.responses.create = Mock(wraps=client.responses.create)
        logger = FakeLangfuse()
        with tempfile.TemporaryDirectory() as directory, Trace({}, logger).run("offline"):
            config = build_index([doc], client, directory, documents_sha256="ab" * 32)
            self.assertEqual(config["chunk_metadata_version"], 1)
            index, chunks, saved = load_index(directory)
            self.assertEqual(saved, read_json(Path(directory) / "config.json"))
            hits = retrieve("예산", client, index, chunks, saved)
            answer = generate_answer("예산", hits, client)
            self.assertEqual(answer["sources"][0]["metadata"]["section_path"], "PRIVATE_SECTION")
            self.assertIn("PRIVATE_SECTION", client.responses.create.call_args.kwargs["input"])
            self.assertEqual(json.loads(json.dumps(answer)), answer)
            # 구형 저장 파일도 계속 읽고 검색할 수 있습니다.
            saved.pop("chunk_metadata_version")
            chunks[0]["metadata"].pop("section_path")
            chunks[0]["metadata"].pop("tables")
            write_json(Path(directory) / "config.json", saved)
            write_json(Path(directory) / "chunks.json", chunks)
            index, chunks, saved = load_index(directory)
            self.assertEqual(retrieve("예산", client, index, chunks, saved)[0]["doc_id"], "a")
        self.assertNotIn("PRIVATE_SECTION", json.dumps(logger.records))


class TableMetadataTests(unittest.TestCase):
    @staticmethod
    def table(rows=20, newline="\n"):
        cells = [(0, 0, 1, 1, "항목🙂"), (0, 1, 1, 1, "PRIVATE_HEADER")]
        cells += [(r, c, 1, 1, f"값{r}-{c}") for r in range(1, rows) for c in range(2)]
        return _number_tables(_render_table(rows, 2, cells)).replace("\n", newline)

    def chunks(self, text, size=80, overlap=15, strategy="structured"):
        doc = {"doc_id": "a", "text": text, "metadata": {"사업명": "예시"}}
        original = copy.deepcopy(doc)
        with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
            chunks = chunk_documents([doc], size, overlap)
        self.assertEqual(original, doc)
        self.assertEqual(json.loads(json.dumps(chunks)), chunks)
        covered = set()
        for chunk in chunks:
            meta = chunk["metadata"]
            start, end = meta["start_char"], meta["end_char"]
            self.assertEqual(chunk["text"], text[start:end])
            self.assertLessEqual(end - start, size)
            covered.update(range(start, end))
            for table in meta["tables"]:
                self.assertLess(table["start_char"], end)
                self.assertGreater(table["end_char"], start)
                if table["header_text"]:
                    self.assertEqual(table["header_text"],
                                     text[table["header_start_char"]:table["header_end_char"]])
        self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))
        return chunks

    def test_long_table_rows_retain_original_header_and_id_in_all_modes(self):
        for newline in ("\n", "\r\n"):
            text = self.table(newline=newline)
            for strategy in ("fixed", "boundary", "structured"):
                with self.subTest(newline=repr(newline), strategy=strategy):
                    chunks = self.chunks(text, strategy=strategy)
                    expected = chunks[0]["metadata"]["tables"][0]
                    self.assertEqual(expected["table_id"], "T1")
                    self.assertEqual((expected["start_char"], expected["end_char"]), (0, len(text)))
                    self.assertIn("PRIVATE_HEADER", expected["header_text"])
                    self.assertTrue(any("PRIVATE_HEADER" not in c["text"] for c in chunks))
                    boundaries = {0, len(text)} | {i + 1 for i, char in enumerate(text) if char == "\n"}
                    for chunk in chunks:
                        self.assertEqual(chunk["metadata"]["tables"], [expected])
                        if strategy == "structured":
                            self.assertIn(chunk["metadata"]["start_char"], boundaries)
                            self.assertIn(chunk["metadata"]["end_char"], boundaries)
                    chunks[0]["metadata"]["tables"][0]["table_id"] = "changed"
                    self.assertEqual(chunks[1]["metadata"]["tables"][0]["table_id"], "T1")

    def test_short_table_is_whole_and_multiple_tables_are_identified(self):
        first = self.table(rows=3)
        second = first.replace("T1", "T2")
        text = "앞\n" + first + "\n" + second + "\n끝"
        chunks = self.chunks(text, size=len(first), overlap=20)
        for table in (first, second):
            self.assertTrue(any(table in c["text"] for c in chunks))
        whole = self.chunks(text, size=len(text), overlap=0)[0]
        self.assertEqual([t["table_id"] for t in whole["metadata"]["tables"]], ["T1", "T2"])

    def test_half_open_intersection_and_plain_text_have_no_phantom_tables(self):
        table = self.table(rows=3)
        size = len(table)
        chunks = self.chunks("앞" * size + table + "뒤" * size, size=size, overlap=0, strategy="fixed")
        self.assertEqual([len(c["metadata"]["tables"]) for c in chunks], [0, 1, 0])
        self.assertEqual(self.chunks("일반 본문")[0]["metadata"]["tables"], [])

    def test_legacy_missing_header_and_invalid_markers(self):
        legacy = self.table(rows=3).replace(":T1", "")
        tables = self.chunks(legacy)[0]["metadata"]["tables"]
        self.assertIsNone(tables[0]["table_id"])
        self.assertIn("PRIVATE_HEADER", tables[0]["header_text"])
        for body in ("| 항목 | 내용 |\n| 예산 | 100 |", "일반 본문", "| a | b |\n| -- | bad |"):
            text = f"<!-- table:T1 -->\n{body}\n<!-- /table:T1 -->"
            entry = self.chunks(text)[0]["metadata"]["tables"][0]
            self.assertEqual(entry["header_text"], "")
            self.assertIsNone(entry["header_start_char"])
            self.assertIsNone(entry["header_end_char"])
        valid = self.table(rows=3)
        for malformed in (valid.replace("/table:T1", "/table:T2"), valid.split("<!-- /table:")[0],
                          "<!-- table:T2 -->" + valid + "<!-- /table:T2 -->"):
            self.assertTrue(all(not c["metadata"]["tables"] for c in self.chunks(malformed)))

    def test_oversized_row_and_tiny_chunks_still_get_header(self):
        text = self.table(rows=3).replace("값1-0", "아주긴값" * 100)
        for size in (1, 7, 80):
            chunks = self.chunks(text, size=size, overlap=0)
            self.assertTrue(all("PRIVATE_HEADER" in c["metadata"]["tables"][0]["header_text"] for c in chunks))

    def test_table_metadata_reaches_generation_without_changing_embeddings_or_trace(self):
        text = self.table()
        doc = {"doc_id": "a", "text": text, "metadata": {"filename": "a.hwp"}}
        client = FakeClient()
        client.embeddings.create = Mock(wraps=client.embeddings.create)
        client.responses.create = Mock(wraps=client.responses.create)
        logger = FakeLangfuse()
        with tempfile.TemporaryDirectory() as directory, Trace({}, logger).run("offline"), \
                patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "structured", "RFP_EMBEDDING_CONTEXT": "none"}):
            config = build_index([doc], client, directory, chunk_size=80, chunk_overlap=15)
            index, chunks, saved = load_index(directory)
            self.assertEqual(config, saved)
            inputs = [text for call in client.embeddings.create.call_args_list for text in call.kwargs["input"]]
            self.assertEqual(inputs, [c["text"] for c in chunks])
            hits = retrieve("예산", client, index, chunks, config)
            answer = generate_answer("예산", hits, client)
            for source in answer["sources"]:
                self.assertEqual(source["metadata"]["tables"][0]["table_id"], "T1")
                self.assertIn("PRIVATE_HEADER", source["metadata"]["tables"][0]["header_text"])
            self.assertIn('"header_text"', client.responses.create.call_args.kwargs["input"])
        self.assertNotIn("PRIVATE_HEADER", json.dumps(logger.records))
