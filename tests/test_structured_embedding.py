"""사업명 문맥 및 구조 청킹: API 없이 입력·원문 좌표·저장·추적 계약 검증."""

import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from embedding import _embedding_input, build_index, chunk_documents, load_index
from evaluation import evaluate
from generation import generate_answer
from observability import Trace
from parsing import write_json
from retrieval import retrieve
from test_embedding import document, response, response_client
from test_observability import FakeLangfuse
from test_pipeline import FakeClient


def table(body="| 항목 | 내용 |\n| 예산 | 100원 |\n", tag="T1"):
    return f"<!-- table:{tag} -->\n{body}<!-- /table:{tag} -->"


class StructuredTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "structured"})
        env.start()
        self.addCleanup(env.stop)

    def assert_contract(self, doc, size, overlap):
        before = copy.deepcopy(doc)
        chunks = chunk_documents([doc], size, overlap)
        text = doc["text"]
        coverage = bytearray(len(text))
        last_start, last_end = -1, 0
        for c in chunks:
            start, end = (c["metadata"][k] for k in ("start_char", "end_char"))
            self.assertTrue(0 <= start < end <= len(text))
            self.assertLessEqual(end - start, size)
            self.assertGreater(start, last_start)
            self.assertGreater(end, last_end)
            self.assertLessEqual(max(0, last_end - start), overlap)
            self.assertEqual(c["text"], text[start:end])
            self.assertTrue(c["text"].strip())
            self.assertEqual(c["doc_id"], doc["doc_id"])
            self.assertRegex(c["chunk_id"], rf"^{doc['doc_id']}:\d+$")
            for key, value in doc["metadata"].items():
                self.assertEqual(c["metadata"][key], value)
            coverage[start:end] = b"\1" * (end - start)
            last_start, last_end = start, end
        self.assertTrue(all(coverage[i] for i, c in enumerate(text) if not c.isspace()))
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        self.assertEqual(json.loads(json.dumps(chunks)), chunks)
        self.assertEqual(doc, before)
        return chunks

    def test_small_tables_are_whole_even_at_limit_and_with_large_overlap(self):
        t = table()
        for prefix in ("", "머리\n", "가" * 90 + "\n"):
            for size in (len(t), len(t) + 10, len(t) * 2):
                for overlap in (0, size // 2, size - 1):
                    with self.subTest(prefix=len(prefix), size=size, overlap=overlap):
                        chunks = self.assert_contract(document(prefix + t + "\n" + t + "\n끝"), size, overlap)
                        for left in (len(prefix), len(prefix) + len(t) + 1):
                            right = left + len(t)
                            self.assertTrue(any(c["metadata"]["start_char"] <= left and
                                                c["metadata"]["end_char"] >= right for c in chunks))
                            for c in chunks:
                                for key in ("start_char", "end_char"):
                                    self.assertFalse(left < c["metadata"][key] < right)

    def test_large_table_preserves_rows_and_markers_without_copying_headers(self):
        t = table("".join(f"| 행{i} | 값{i} |\n" for i in range(20)))
        chunks = self.assert_contract(document(t), 70, 15)
        boundaries = {0, len(t)} | {i + 1 for i, c in enumerate(t) if c == "\n"}
        for c in chunks:
            self.assertIn(c["metadata"]["start_char"], boundaries)
            self.assertIn(c["metadata"]["end_char"], boundaries)

    def test_fitting_table_stays_with_preface_at_paragraph_boundary(self):
        for newline in ("\n", "\r\n"):
            prefix = "예시 사업🙂e\u0301 " + "설명" * 80 + newline * 2
            t = table().replace("\n", newline)
            text = prefix + t + newline + "후속 설명" * 100
            for spare in (0, 10):
                size = len(prefix + t) + spare
                for overlap in (0, 15, size - 1):
                    with self.subTest(newline=repr(newline), spare=spare, overlap=overlap):
                        chunks = self.assert_contract(document(text), size, overlap)
                        self.assertIn(prefix + t, chunks[0]["text"])
                        if overlap < len(prefix):
                            self.assertEqual(chunks[0]["metadata"]["end_char"], len(prefix + t))
                        self.assertEqual(chunks[0]["metadata"]["tables"][0]["table_id"], "T1")

    def test_table_one_character_over_limit_still_moves_whole(self):
        prefix = "설명" * 80 + "\n\n"
        t = table()
        chunks = self.assert_contract(document(prefix + t + "\n" + "뒤" * 300),
                                      len(prefix + t) - 1, 15)
        self.assertEqual(chunks[0]["text"], prefix)
        self.assertTrue(any(t in c["text"] for c in chunks[1:]))

    def test_fitting_table_does_not_override_actual_section_boundary(self):
        prefix = "앞 문맥" * 40 + "\n\n"
        section = "Ⅱ 새 제목\n새 본문\n\n"
        t = table()
        doc = {**document(prefix + section + t + "\n" + "뒤" * 300),
               "sections": [[len(prefix), "Ⅱ 새 제목"]]}
        chunks = self.assert_contract(doc, len(prefix + section + t), 15)
        self.assertEqual(chunks[0]["text"], prefix)
        self.assertTrue(any(t in c["text"] for c in chunks[1:]))

    def test_adjacent_tables_and_invalid_markers_do_not_expand_past_limit(self):
        prefix = "설명" * 80 + "\n\n"
        first, second = table(), table(tag="T2")
        size = len(prefix + first)
        chunks = self.assert_contract(document(prefix + first + second + "뒤" * 300), size, 15)
        self.assertEqual(chunks[0]["text"], prefix + first)
        self.assertTrue(any(second in c["text"] for c in chunks[1:]))
        for invalid in (first.replace("/table:T1", "/table:T2"),
                        "<!-- table:T2 -->" + first + "<!-- /table:T2 -->"):
            chunks = self.assert_contract(document(prefix + invalid + "뒤" * 300),
                                          len(prefix + invalid), 15)
            self.assertEqual(chunks[0]["text"], prefix)
            self.assertFalse(chunks[0]["metadata"]["tables"])

    def test_preface_table_change_is_structured_only(self):
        prefix = "설명" * 80 + "\n\n"
        text = prefix + table() + "\n" + "뒤" * 300
        size = len(prefix + table()) + 10
        for mode, expected in (("fixed", size), ("boundary", len(prefix)),
                               ("structured", len(prefix + table()))):
            with self.subTest(mode=mode), patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": mode}):
                self.assertEqual(chunk_documents([document(text)], size, 15)[0]["metadata"]["end_char"],
                                 expected)

    def test_oversized_row_tiny_chunks_unicode_and_whitespace_terminate(self):
        texts = [table("| " + "한글🙂e\u0301" * 80 + " |\n"), "", " \t\r\n" * 30,
                 "가" * 83, "앞\r\n\n다음. 끝! " * 10]
        for text in texts:
            for size in (1, 2, 7, 40, 100):
                for overlap in sorted({0, size // 2, size - 1}):
                    with self.subTest(size=size, overlap=overlap, length=len(text)):
                        self.assert_contract(document(text), size, overlap)

    def test_section_boundary_and_title_stay_with_following_body(self):
        text = "가" * 65 + "\nⅡ 새 제목\n" + "나" * 25 + "\n" + "다" * 150
        pos = text.index("Ⅱ")
        doc = {**document(text), "sections": [[pos, "Ⅱ 새 제목"]]}
        chunks = self.assert_contract(doc, 100, 15)
        self.assertEqual(chunks[0]["metadata"]["end_char"], pos)
        self.assertTrue(any("Ⅱ 새 제목\n" + "나" * 25 in c["text"] for c in chunks))
        title_end = text.index("\n", pos) + 1
        self.assertTrue(all(c["metadata"]["end_char"] != title_end for c in chunks))

    def test_toc_preface_keeps_first_overview_facts_with_document_title(self):
        for toc in ("목 차", "목차", "< 목 차 >"):
            for newline in ("\n", "\r\n"):
                with self.subTest(toc=toc, newline=repr(newline)):
                    preface = newline.join(["사업 표지", toc, "목록 " * 175, "", ""])
                    overview = newline.join(["Ⅰ 사업 개요", "사업명: 예시 시스템", "기간: 180일",
                                             "예산: 150,000,000원", "", ""])
                    text = preface + overview + "후속 설명 " * 200
                    doc = {**document(text), "sections": [[len(preface), "Ⅰ 사업 개요"]]}
                    chunks = self.assert_contract(doc, 1000, 150)
                    self.assertTrue(chunks[0]["text"].startswith("사업 표지"))
                    self.assertIn("기간: 180일", chunks[0]["text"])
                    self.assertIn("예산: 150,000,000원", chunks[0]["text"])

    def test_inline_toc_word_does_not_disable_normal_section_boundary(self):
        preface = "본문에서 목차를 설명 " + "가" * 60 + "\n"
        text = preface + "Ⅱ 새 제목\n다음 본문\n" + "나" * 150
        doc = {**document(text), "sections": [[len(preface), "Ⅱ 새 제목"]]}
        chunks = self.assert_contract(doc, 100, 15)
        self.assertEqual(chunks[0]["metadata"]["end_char"], len(preface))

    def test_toc_overview_preserves_tables_size_and_overlap_limits(self):
        preface = "사업 표지\n목 차\n" + "목록 " * 175 + "\n\n"
        overview = "Ⅰ 사업 개요\n사업명: 예시\n" + table() + "\n\n"
        text = preface + overview + "후속 설명 " * 200
        doc = {**document(text), "sections": [[len(preface), "Ⅰ 사업 개요"]]}
        for size in (1000, len(preface) + 10, len(preface) + 40):
            for overlap in (0, 150, size - 1):
                with self.subTest(size=size, overlap=overlap):
                    chunks = self.assert_contract(doc, size, overlap)
                    self.assertTrue(any(table() in c["text"] for c in chunks))

    def test_structured_revision_is_saved_without_changing_other_strategy_versions(self):
        for strategy, version in (("fixed", 1), ("boundary", 1), ("structured", 3)):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy,
                                            "RFP_EMBEDDING_CONTEXT": "none"}):
                config = build_index([document("예산 100원")], FakeClient(), directory)
                self.assertEqual(config["chunking_version"], version)
                self.assertEqual(load_index(directory)[2], config)

    def test_structured_v2_and_v3_are_selectable_and_keep_chunk_contract(self):
        prefix = "가" * 70 + "\n\n"
        text = prefix + table() + "\n" + "후속 설명 " * 30
        doc = document(text)
        results = {}
        for version in ("2", "3"):
            with self.subTest(version=version), patch.dict(os.environ, {
                    "RFP_CHUNKING_STRATEGY": "structured",
                    "RFP_STRUCTURED_CHUNKING_VERSION": version}):
                results[version] = self.assert_contract(doc, 140, 10)
                with tempfile.TemporaryDirectory() as directory:
                    config = build_index([doc], FakeClient(), directory)
                    self.assertEqual(config["chunking_version"], int(version))
                    self.assertEqual(load_index(directory)[1], chunk_documents([doc], 1000, 150))
        self.assertNotIn(table(), results["2"][0]["text"])
        self.assertIn(table(), results["3"][0]["text"])

    def test_invalid_structured_version_fails_before_embedding_or_writes(self):
        for value in ("", "1", "4", "v3"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "structured",
                                            "RFP_STRUCTURED_CHUNKING_VERSION": value}):
                client = response_client(None)
                target = Path(directory) / "index"
                with self.assertRaisesRegex(ValueError, "RFP_STRUCTURED_CHUNKING_VERSION"):
                    build_index([document("본문")], client, target)
                client.embeddings.create.assert_not_called()
                self.assertFalse(target.exists())

    def test_adjacent_titles_and_title_before_table(self):
        text = "Ⅰ 제목\n1. 소제목\n본문입니다\n" + table() + "\n후속" * 20
        doc = {**document(text), "sections": [[0, "Ⅰ 제목"], [5, "소제목"]]}
        for overlap in (0, 20, 79):
            self.assert_contract(doc, 80, overlap)
        doc = {**document("Ⅰ 제목\n" + table()), "sections": [[0, "Ⅰ 제목"]]}
        chunks = self.assert_contract(doc, len(table()), 10)
        self.assertTrue(any(c["text"] == table() for c in chunks))

    def test_missing_or_invalid_sections_are_ignored(self):
        doc = document("첫 문단\n\n" + "둘째 문단\n" * 20)
        expected = chunk_documents([doc], 30, 5)
        invalid = [None, {}, "bad", [[-1, "제목"], [99999, "제목"], [True, "제목"],
                                    [1, "중간 위치"], [0, None], [0, ""], [0], "bad"]]
        for sections in invalid:
            with self.subTest(sections=sections):
                self.assertEqual(chunk_documents([{**doc, "sections": sections}], 30, 5), expected)

    def test_malformed_nested_and_legacy_table_markers(self):
        malformed = ["<!-- table:T1 -->\n내용\n<!-- /table:T2 -->",
                     "<!-- table:T1 -->\n내용", "<!-- /table:T1 -->\n내용",
                     "<!-- table:T1 -->" + table(tag="T2") + "<!-- /table:T1 -->"]
        for text in malformed:
            self.assert_contract(document(text * 3), 40, 10)
        legacy = "<!-- table -->\n| 가 | 나 |\n<!-- /table -->"
        chunks = self.assert_contract(document("앞" * 20 + legacy + "뒤" * 20), len(legacy), 20)
        self.assertTrue(any(c["text"] == legacy for c in chunks))

    def test_deterministic_mixed_documents_cover_every_character(self):
        rng = random.Random(20261006)
        parts = ["제목\n", "문장. 다음!\n\n", "가" * 53, "\t \r\n", table(), table("긴행" * 100)]
        for _ in range(80):
            text = "".join(rng.choices(parts, k=6))
            size = rng.randint(1, 180)
            self.assert_contract(document(text), size, rng.randrange(size))


class ProjectEmbeddingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"RFP_EMBEDDING_CONTEXT": "project",
                                     "RFP_CHUNKING_STRATEGY": "structured"})
        env.start()
        self.addCleanup(env.stop)

    def test_context_normalization_missing_values_and_saved_text(self):
        docs = [document("원문🙂e\u0301", str(i)) for i in range(5)]
        metas = [{"사업명": " 가\n 사업\t", "발주 기관": " 기관\r\n 가 "},
                 {"사업명": None, "발주 기관": "기관만"},
                 {"사업명": 123, "발주 기관": []},
                 {"사업명": "사업만", "발주 기관": " \n"}, {}]
        for doc, meta in zip(docs, metas):
            doc["metadata"] = meta
        before = copy.deepcopy(docs)
        client = response_client(response([[1., 2.]] * len(docs)))
        with tempfile.TemporaryDirectory() as directory:
            config = build_index(docs, client, directory)
            _, chunks, saved = load_index(directory)
            self.assertEqual(config, saved)
            self.assertEqual(config["embedding_context"], "project")
            self.assertEqual(config["embedding_context_version"], 1)
            self.assertEqual(chunks, chunk_documents(docs))
        inputs = client.embeddings.create.call_args.kwargs["input"]
        body = docs[0]["text"]
        self.assertEqual(inputs, ["사업명: 가 사업\n발주 기관: 기관 가\n\n" + body,
                                  "발주 기관: 기관만\n\n" + body, body,
                                  "사업명: 사업만\n\n" + body, body])
        self.assertEqual(docs, before)

    def test_default_and_explicit_none_preserve_raw_embedding_inputs(self):
        doc = document("본문")
        doc["metadata"]["사업명"] = "사업"
        for explicit in (True, False):
            with patch.dict(os.environ):
                if explicit:
                    os.environ["RFP_EMBEDDING_CONTEXT"] = "none"
                else:
                    os.environ.pop("RFP_EMBEDDING_CONTEXT")
                client = response_client(response([[1., 2.]]))
                with tempfile.TemporaryDirectory() as directory:
                    config = build_index([doc], client, directory)
                self.assertEqual(config["embedding_context"], "none")
                self.assertEqual(client.embeddings.create.call_args.kwargs["input"], ["본문"])

    def test_invalid_context_fails_before_api_and_writes(self):
        for value in ("", "PROJECT", "unknown"):
            with patch.dict(os.environ, {"RFP_EMBEDDING_CONTEXT": value}), \
                    tempfile.TemporaryDirectory() as directory:
                client = response_client(None)
                target = Path(directory) / "index"
                with self.assertRaisesRegex(ValueError, "RFP_EMBEDDING_CONTEXT"):
                    build_index([document("내용")], client, target)
                client.embeddings.create.assert_not_called()
                self.assertFalse(target.exists())

    def test_context_batches_vector_order_and_trace_privacy(self):
        docs = [document(f"PRIVATE_BODY_{i}", str(i)) for i in range(65)]
        for i, doc in enumerate(docs):
            doc["metadata"]["사업명"] = f"PRIVATE_PROJECT_{i}"
        client = response_client(None)

        def create(model, input):
            result = response([[int(t.rsplit("_", 1)[1]) + 1., 1.] for t in input])
            result.data.reverse()
            return result

        client.embeddings.create.side_effect = create
        logger = FakeLangfuse()
        with tempfile.TemporaryDirectory() as directory, Trace({}, logger).run("test"):
            config = build_index(docs, client, directory)
            index, chunks, _ = load_index(directory)
            self.assertEqual([c["doc_id"] for c in chunks], [str(i) for i in range(65)])
            expected = np.asarray([[i + 1., 1.] for i in range(65)], dtype="float32")
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            np.testing.assert_allclose(index.reconstruct_n(0, 65), expected, atol=1e-7)
        calls = client.embeddings.create.call_args_list
        self.assertEqual([len(c.kwargs["input"]) for c in calls], [32, 32, 1])
        for i, text in enumerate(t for call in calls for t in call.kwargs["input"]):
            self.assertEqual(text, f"사업명: PRIVATE_PROJECT_{i}\n발주 기관: PRIVATE_AGENCY\n\nPRIVATE_BODY_{i}")
        for private in ("PRIVATE_BODY", "PRIVATE_PROJECT", "PRIVATE_FILENAME", "PRIVATE_AGENCY"):
            self.assertNotIn(private, json.dumps(logger.records))
            self.assertNotIn(private, json.dumps(config))

    def test_pipeline_query_is_unmodified_and_legacy_config_loads(self):
        doc = document("예산 100원", "a")
        doc["metadata"]["사업명"] = "PRIVATE_PROJECT"
        client = FakeClient()
        client.embeddings.create = Mock(wraps=client.embed)
        with tempfile.TemporaryDirectory() as directory:
            config = build_index([doc], client, directory, model="saved-model")
            for legacy in (False, True):
                if legacy:
                    for key in ("chunking_strategy", "chunking_version", "embedding_context", "embedding_context_version"):
                        config.pop(key)
                    write_json(Path(directory) / "config.json", config)
                index, chunks, saved = load_index(directory)
                self.assertEqual(saved, config)
                hits = retrieve("예산", client, index, chunks, saved, filters={"사업명": "PRIVATE_PROJECT"})
                self.assertEqual(client.embeddings.create.call_args.kwargs, {"model": "saved-model", "input": ["예산"]})
                answer = generate_answer("예산", hits, client)
                result = evaluate([{"question": "예산", "expected_doc_ids": ["a"],
                                    "expected_keywords": ["100원"]}], lambda q, f: answer)
                self.assertEqual(result["summary"]["recall_at_k"], 1.)
                self.assertEqual(result["summary"]["keyword_coverage"], 1.)
                self.assertEqual(answer["sources"][0]["text"], doc["text"])
                self.assertEqual(answer["sources"][0]["citation"], 1)
                self.assertEqual(json.loads(json.dumps(answer)), answer)

    def test_failed_context_build_preserves_existing_index(self):
        docs = [document(str(i), str(i)) for i in range(33)]
        with tempfile.TemporaryDirectory() as directory:
            build_index([document("예산")], FakeClient(), directory)
            before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            client = response_client(None)
            client.embeddings.create.side_effect = [response([[1., 2.]] * 32), RuntimeError("failure")]
            with self.assertRaises(RuntimeError):
                build_index(docs, client, directory)
            self.assertEqual({p.name: p.read_bytes() for p in Path(directory).iterdir()}, before)


class BlendedProjectTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"RFP_EMBEDDING_CONTEXT": "project_blend",
                                     "RFP_CHUNKING_STRATEGY": "structured"})
        env.start()
        self.addCleanup(env.stop)

    def test_weighted_vectors_preserve_text_and_skip_missing_context(self):
        docs = [document("원문 A", "a"), document("원문 B", "b")]
        docs[0]["metadata"] = {"사업명": "사업 A"}
        docs[1]["metadata"] = {}
        before = copy.deepcopy(docs)
        client = response_client(None)
        client.embeddings.create.side_effect = [response([[3., 0.], [0., 4.]]), response([[0., 5.]])]
        with tempfile.TemporaryDirectory() as directory:
            config = build_index(docs, client, directory)
            index, chunks, saved = load_index(directory)
            self.assertEqual(config, saved)
            self.assertEqual(saved["embedding_context"], "project_blend")
            self.assertEqual(saved["embedding_context_version"], 1)
            expected = np.asarray([[.8, .2], [0., 1.]], dtype="float32")
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            np.testing.assert_allclose(index.reconstruct_n(0, 2), expected, atol=1e-7)
            self.assertEqual(chunks, chunk_documents(docs))
        self.assertEqual([c.kwargs["input"] for c in client.embeddings.create.call_args_list],
                         [["원문 A", "원문 B"], ["사업명: 사업 A\n\n원문 A"]])
        self.assertEqual(docs, before)

    def test_no_context_needs_only_body_request(self):
        doc = document("원문")
        doc["metadata"] = {"사업명": "  ", "발주 기관": None}
        client = response_client(response([[3., 4.]]))
        with tempfile.TemporaryDirectory() as directory:
            build_index([doc], client, directory)
            index, _, _ = load_index(directory)
            np.testing.assert_allclose(index.reconstruct(0), [.6, .8], atol=1e-7)
        client.embeddings.create.assert_called_once_with(model="text-embedding-3-small", input=["원문"])

    def test_context_failure_or_dimension_mismatch_preserves_index(self):
        doc = document("예산")
        for failure in (RuntimeError("failure"), response([[1., 0., 0.]])):
            with self.subTest(failure=type(failure).__name__), tempfile.TemporaryDirectory() as directory:
                build_index([doc], FakeClient(), directory)
                before = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
                client = response_client(None)
                client.embeddings.create.side_effect = [response([[1., 0.]]), failure]
                with self.assertRaises((RuntimeError, ValueError)):
                    build_index([doc], client, directory)
                self.assertEqual({p.name: p.read_bytes() for p in Path(directory).iterdir()}, before)

    def test_mixed_metadata_batches_order_and_trace_privacy(self):
        docs = [document(f"PRIVATE_BODY_{i}", str(i)) for i in range(65)]
        for i, doc in enumerate(docs):
            doc["metadata"] = {"사업명": f"PRIVATE_PROJECT_{i}"} if i % 2 == 0 else {}
        client = response_client(None)

        def create(model, input):
            rows = []
            for text in input:
                i = int(text.rsplit("_", 1)[1])
                rows.append([1., i + 1.] if text.startswith("사업명:") else [i + 1., 1.])
            result = response(rows)
            result.data.reverse()
            return result

        client.embeddings.create.side_effect = create
        logger = FakeLangfuse()
        with tempfile.TemporaryDirectory() as directory, Trace({}, logger).run("blend-offline"):
            build_index(docs, client, directory)
            index, chunks, _ = load_index(directory)
            self.assertEqual([c["doc_id"] for c in chunks], [str(i) for i in range(65)])
            expected = np.asarray([[i + 1., 1.] for i in range(65)], dtype="float32")
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            context = expected[::2, ::-1].copy()
            expected[::2] = .8 * expected[::2] + .2 * context
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            np.testing.assert_allclose(index.reconstruct_n(0, 65), expected, atol=2e-7)
        self.assertEqual([len(c.kwargs["input"]) for c in client.embeddings.create.call_args_list],
                         [32, 16, 32, 16, 1, 1])
        for private in ("PRIVATE_BODY", "PRIVATE_PROJECT"):
            self.assertNotIn(private, json.dumps(logger.records))

    def test_query_and_answer_contract_with_blended_index(self):
        doc = document("예산 100원", "a")
        doc["metadata"]["사업명"] = "사업 A"
        client = FakeClient()
        client.embeddings.create = Mock(wraps=client.embed)
        with tempfile.TemporaryDirectory() as directory:
            build_index([doc], client, directory, model="saved-model")
            index, chunks, config = load_index(directory)
            hits = retrieve("예산", client, index, chunks, config, filters={"사업명": "사업 A"})
            self.assertEqual(client.embeddings.create.call_args.kwargs, {"model": "saved-model", "input": ["예산"]})
            answer = generate_answer("예산", hits, client)
            result = evaluate([{"question": "예산", "expected_doc_ids": ["a"], "expected_keywords": ["100원"]}],
                              lambda q, f: answer)
            self.assertEqual(result["summary"]["recall_at_k"], 1.)
            self.assertEqual(result["summary"]["keyword_coverage"], 1.)
            self.assertEqual(answer["sources"][0]["text"], doc["text"])
            self.assertEqual(json.loads(json.dumps(answer)), answer)


class TableEmbeddingContextTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"RFP_EMBEDDING_CONTEXT": "table",
                                     "RFP_CHUNKING_STRATEGY": "structured"})
        env.start()
        self.addCleanup(env.stop)

    def test_table_context_adds_section_and_only_missing_headers_without_changing_chunks(self):
        text = ("Ⅰ 예산 편성\n\n" + table(
            "| 구분 | 예산액 |\n|---|---|\n" + "".join(f"| 세부 항목 {i} | {i}원 |\n" for i in range(20))))
        doc = {**document(text), "sections": [[0, "Ⅰ 예산 편성"]]}
        client = response_client(None)

        def create(model, input):
            result = response([[i + 1., 1.] for i in range(len(input))])
            result.data.reverse()
            return result

        client.embeddings.create.side_effect = create
        with tempfile.TemporaryDirectory() as directory:
            config = build_index([doc], client, directory, chunk_size=100, chunk_overlap=15)
            index, chunks, saved = load_index(directory)
            self.assertEqual(config, saved)
            self.assertEqual(saved["embedding_context"], "table")
            self.assertEqual(saved["embedding_context_version"], 2)
            self.assertEqual([chunk["text"] for chunk in chunks],
                             [text[c["metadata"]["start_char"]:c["metadata"]["end_char"]] for c in chunks])
            np.testing.assert_allclose(np.linalg.norm(index.reconstruct_n(0, len(chunks)), axis=1), 1., atol=1e-6)

        inputs = [value for call in client.embeddings.create.call_args_list for value in call.kwargs["input"]]
        self.assertEqual(len(inputs), len(chunks))
        self.assertTrue(any("절: Ⅰ 예산 편성" in value for value in inputs))
        continuation = next(c for c in chunks if c["metadata"]["tables"]
                            and c["metadata"]["tables"][0]["header_text"]
                            and c["metadata"]["tables"][0]["header_text"] not in c["text"])
        contextual = inputs[chunks.index(continuation)]
        self.assertIn("표 헤더:\n" + continuation["metadata"]["tables"][0]["header_text"], contextual)
        self.assertTrue(contextual.endswith("\n\n" + continuation["text"]))
        self.assertNotIn("표 헤더:", inputs[0])
        self.assertEqual(_embedding_input({"text": "일반 본문", "metadata": {}}, "table"), "일반 본문")

    def test_table_context_handles_crlf_and_legacy_metadata_safely(self):
        header = "| 구분 | 금액 |\r\n|---|---|"
        chunk = {"text": "| 세부 | 100원 |", "metadata": {"section_path": " Ⅰ   예산 ",
                  "tables": [{"header_text": header}, {"header_text": header}, {}]}}
        actual = _embedding_input(chunk, "table")
        self.assertEqual(actual, "절: Ⅰ 예산\n표 헤더:\n" + header.replace("\r\n", "\n")
                         + "\n\n" + chunk["text"])


if __name__ == "__main__":
    unittest.main()
