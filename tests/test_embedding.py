"""외부 통신 없이 청킹·임베딩 계약, 실패 격리, 파이프라인 연결을 검증합니다."""

import copy
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# discover 외의 실행 방식에서도 프로젝트 모듈과 테스트 보조 모듈을 찾도록 합니다.
TESTS_DIR = Path(__file__).resolve().parent
for path in (TESTS_DIR, TESTS_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

from embedding import build_index, chunk_documents, embed_texts, load_index
from evaluation import evaluate
from generation import generate_answer
from observability import Trace
from parsing import read_json, write_json
from retrieval import retrieve
from test_observability import FakeLangfuse
from test_pipeline import FakeClient


def document(text, doc_id="doc"):
    return {"doc_id": doc_id, "text": text, "metadata": {
        "filename": "PRIVATE_FILENAME.hwp", "source": "files/PRIVATE_FILENAME.hwp",
        "발주 기관": "PRIVATE_AGENCY",
    }}


def response(rows, indices=None):
    if indices is None:
        indices = range(len(rows))
    return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=row)
                                 for i, row in zip(indices, rows)])


def response_client(result):
    return SimpleNamespace(embeddings=SimpleNamespace(create=Mock(return_value=result)))


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "boundary"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_boundary_priority_and_latest_of_same_kind(self):
        cases = [
            ("가" * 9 + "\n\n" + "나" * 5 + "\n" + "다" * 30, 11),
            ("가" * 7 + "\n\n" + "나" * 5 + "\n\n" + "다" * 30, 16),
            ("가" * 10 + "\n" + "나" * 5 + "\n" + "다" * 30, 17),
            ("가" * 10 + "\n" + "나" * 4 + ". " + "다" * 30, 11),
            ("가" * 9 + ". " + "나" * 30, 11),
            ("가" * 9 + "。 " + "나" * 30, 11),
            ("가" * 9 + "\r\n \t\r\n" + "나" * 30, 15),
            ("가" * 18 + "\n\n" + "나" * 30, 20),
            ("가" * 9 + "\n\n\n" + "나" * 30, 12),
        ]
        for text, end in cases:
            with self.subTest(end=end, text=text):
                chunks = chunk_documents([document(text)], 20, 3)
                self.assertEqual(chunks[0]["metadata"]["end_char"], end)
                self.assertEqual(chunks[1]["metadata"]["start_char"], end - 3)

    def test_short_boundary_is_ignored_and_overlap_makes_progress(self):
        text = "가" * 9 + "\n\n" + "나" * 5 + "\n" + "다" * 30
        chunks = chunk_documents([document(text)], 20, 15)
        self.assertEqual(chunks[0]["metadata"]["end_char"], 17)
        self.assertEqual(chunks[1]["metadata"]["start_char"], 2)
        short = "가\n\n" + "나" * 40
        self.assertEqual(chunk_documents([document(short)], 20, 3)[0]["metadata"]["end_char"], 20)

    def test_unbroken_text_falls_back_to_fixed(self):
        chunks = chunk_documents([document("abcdefghijk")], 5, 2)
        self.assertEqual([c["text"] for c in chunks], ["abcde", "defgh", "ghijk"])

    def test_short_and_exact_size_document_stays_whole(self):
        for text in ("한글\n\n짧음", "가" * 18 + "\n\n"):
            chunks = chunk_documents([document(text)], 20, 3)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0]["text"], text)

    def test_empty_and_whitespace_documents(self):
        self.assertEqual(chunk_documents([]), [])
        for strategy in ("fixed", "boundary"):
            with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
                self.assertEqual(chunk_documents([document(""), document(" \t\r\n" * 20)], 5, 2), [])

    def test_offsets_coverage_termination_and_metadata_for_both_strategies(self):
        texts = ["a", "한글🙂e\u0301\r\n\n문장. 다음! 끝? " * 3,
                 " " * 30 + "본문" + " " * 30, "가" * 71]
        for strategy in ("fixed", "boundary"):
            with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
                for text in texts:
                    original = document(text)
                    before = copy.deepcopy(original)
                    for size in (1, 2, 5, 20):
                        for overlap in sorted({0, size // 2, size - 1}):
                            with self.subTest(strategy=strategy, size=size, overlap=overlap):
                                chunks = chunk_documents([original], size, overlap)
                                covered, starts, ids = set(), [], set()
                                self.assertLessEqual(len(chunks), len(text))
                                for chunk in chunks:
                                    meta = chunk["metadata"]
                                    start, end = meta["start_char"], meta["end_char"]
                                    self.assertTrue(0 <= start < end <= len(text))
                                    self.assertLessEqual(end - start, size)
                                    self.assertEqual(chunk["text"], text[start:end])
                                    self.assertTrue(chunk["text"].strip())
                                    self.assertEqual(chunk["doc_id"], original["doc_id"])
                                    self.assertRegex(chunk["chunk_id"], r"^doc:\d+$")
                                    for key, value in original["metadata"].items():
                                        self.assertEqual(meta[key], value)
                                    covered.update(range(start, end))
                                    starts.append(start)
                                    ids.add(chunk["chunk_id"])
                                self.assertEqual(len(ids), len(chunks))
                                self.assertEqual(starts, sorted(set(starts)))
                                self.assertTrue({i for i, c in enumerate(text) if not c.isspace()} <= covered)
                                self.assertEqual(json.loads(json.dumps(chunks)), chunks)
                                self.assertEqual(original, before)

    def test_fixed_reproduces_original_numbering_and_slices(self):
        # 기존 고정 분할 규약: 공백 구간을 건너뛰어도 순번은 소비합니다.
        with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "fixed"}):
            chunks = chunk_documents([document("     abcdefghijk")], 5, 0)
            self.assertEqual([(c["chunk_id"], c["text"]) for c in chunks],
                             [("doc:1", "abcde"), ("doc:2", "fghij"), ("doc:3", "k")])
            explicit = chunk_documents([document("가나다\n\n라마바" * 3)], 10, 2)
            os.environ.pop("RFP_CHUNKING_STRATEGY")
            self.assertEqual(chunk_documents([document("가나다\n\n라마바" * 3)], 10, 2), explicit)

    def test_multiple_documents_keep_order_ids_and_independent_metadata(self):
        documents = [document("가나다라마.\n\n" * 6, "a"), document("가나다라마.\n\n" * 6, "b")]
        before = copy.deepcopy(documents)

        def positions(chunks, doc_id):
            return [(c["chunk_id"].split(":", 1)[1], c["text"], c["metadata"]["start_char"],
                     c["metadata"]["end_char"]) for c in chunks if c["doc_id"] == doc_id]

        for strategy in ("fixed", "boundary"):
            with self.subTest(strategy=strategy), patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
                chunks = chunk_documents(documents, 20, 3)
                doc_ids = [c["doc_id"] for c in chunks]
                self.assertEqual(set(doc_ids), {"a", "b"})
                self.assertEqual(doc_ids, sorted(doc_ids))
                self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
                for chunk in chunks:
                    self.assertTrue(chunk["chunk_id"].startswith(chunk["doc_id"] + ":"))
                # 같은 본문이면 문서 ID만 다르고 순번·위치는 같아야 합니다.
                self.assertEqual(positions(chunks, "a"), positions(chunks, "b"))
                chunks[0]["metadata"]["발주 기관"] = "CHANGED"
                self.assertTrue(all(c["metadata"]["발주 기관"] == "PRIVATE_AGENCY" for c in chunks[1:]))
                self.assertEqual(documents, before)

    def test_invalid_settings_fail(self):
        for size, overlap in ((0, 0), (-1, 0), (5, -1), (5, 5), (5, 6)):
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                chunk_documents([document("a")], size, overlap)
        for strategy in ("", "BOUNDARY", "unknown"):
            with patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
                with self.assertRaisesRegex(ValueError, "RFP_CHUNKING_STRATEGY"):
                    chunk_documents([])


class EmbeddingTests(unittest.TestCase):
    def test_reverse_response_is_normalized_float32_in_input_order(self):
        client = response_client(response([[0., 7.], [3., 4.]], [1, 0]))
        vectors = embed_texts(["a", "b"], client, "test-model")
        self.assertEqual(vectors.dtype, np.dtype("float32"))
        self.assertTrue(vectors.flags.c_contiguous)
        np.testing.assert_allclose(vectors, [[.6, .8], [0., 1.]], atol=1e-7)
        client.embeddings.create.assert_called_once_with(model="test-model", input=["a", "b"])

    def test_extreme_finite_vectors_normalize_without_overflow_or_underflow(self):
        vectors = embed_texts(["a", "b"], response_client(response([
            [1e30, -1e30], [1e-30, 1e-30]])), "test-model")
        self.assertTrue(np.isfinite(vectors).all())
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), [1., 1.], atol=1e-7)

    def test_empty_input_does_not_call_api(self):
        client = response_client(None)
        with self.assertRaises(ValueError):
            embed_texts([], client, "test-model")
        client.embeddings.create.assert_not_called()

    def test_missing_duplicate_out_of_range_and_non_integer_indices(self):
        for indices in ([], [0], [0, 0], [0, 2], [-1, 0], [0, 1, 2], [0, True], [0, 1.0], [0, "1"]):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "index"):
                embed_texts(["a", "b"], response_client(response([[1., 2.]] * len(indices), indices)), "test")

    def test_bad_vector_shapes_values_and_zero_norm(self):
        for rows in ([[], []], [[1.], [1., 2.]], [1., 2.], [[[1.]], [[2.]]],
                     [[float("nan"), 1.], [1., 2.]], [[float("inf"), 1.], [1., 2.]],
                     [[0., 0.], [1., 2.]], [["bad", 1.], [1., 2.]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                embed_texts(["a", "b"], response_client(response(rows)), "test")


class IndexContractTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "fixed"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_public_signatures_are_unchanged(self):
        expected = {
            chunk_documents: "(documents, chunk_size=1000, chunk_overlap=150)",
            embed_texts: "(texts, client, model)",
            build_index: "(documents, client, index_dir='indexes', model='text-embedding-3-small', chunk_size=1000, chunk_overlap=150)",
            load_index: "(index_dir='indexes')",
        }
        for fn, signature in expected.items():
            self.assertEqual(str(inspect.signature(fn)), signature)

    def test_multiple_batches_keep_global_vector_chunk_order_and_trace_privacy(self):
        client = response_client(None)

        def create(model, input):
            rows = [[float(text.rsplit("_", 1)[1]) + 1, 1.] for text in input]
            result = response(rows)
            result.data.reverse()
            result.usage = SimpleNamespace(prompt_tokens=len(input) * 7, total_tokens=len(input) * 7)
            return result

        client.embeddings.create.side_effect = create
        documents = [document(f"PRIVATE_DOCUMENT_{i}", str(i)) for i in range(65)]
        logger = FakeLangfuse()
        trace = Trace({}, logger)
        with tempfile.TemporaryDirectory() as directory, trace.run("offline-test"):
            config = build_index(documents, client, directory, model="test-model")
            index, chunks, loaded = load_index(directory)
            self.assertEqual(config, loaded)
            self.assertEqual(config["chunking_strategy"], "fixed")
            self.assertEqual(config["chunking_version"], 1)
            self.assertEqual([c["doc_id"] for c in chunks], [str(i) for i in range(65)])
            expected = np.array([[i + 1., 1.] for i in range(65)], dtype="float32")
            expected /= np.linalg.norm(expected, axis=1, keepdims=True)
            np.testing.assert_allclose(index.reconstruct_n(0, 65), expected, atol=1e-7)
            self.assertEqual({p.name for p in Path(directory).iterdir()},
                             {"index.faiss", "chunks.json", "config.json"})
        self.assertEqual([len(c.kwargs["input"]) for c in client.embeddings.create.call_args_list], [32, 32, 1])
        self.assertEqual(trace.usage["test-model"], {"input": 455, "total": 455})
        records = json.dumps(logger.records)
        for private in ("PRIVATE_DOCUMENT", "PRIVATE_FILENAME", "PRIVATE_AGENCY"):
            self.assertNotIn(private, records)
        self.assertEqual([r["model_parameters"]["batch_size"] for r in logger.records if r["name"] == "embedding"], [32, 32, 1])
        self.assertTrue(logger.flushed)

    def test_batch_failure_and_dimension_mismatch_do_not_touch_saved_index(self):
        documents = [document(str(i), str(i)) for i in range(33)]
        for failure in (RuntimeError("PRIVATE_FAILURE"), response([[1., 0., 0.]]),
                        response([[0., 0.]])):
            for existing in (False, True):
                with self.subTest(failure=type(failure).__name__, existing=existing):
                    with tempfile.TemporaryDirectory() as directory:
                        target = Path(directory) / "index"
                        if existing:
                            build_index([document("예산")], FakeClient(), target)
                        before = {p.name: p.read_bytes() for p in target.glob("*")}
                        client = response_client(None)
                        client.embeddings.create.side_effect = [response([[1., 0.]] * 32), failure]
                        with self.assertRaises((ValueError, RuntimeError)):
                            build_index(documents, client, target)
                        self.assertEqual({p.name: p.read_bytes() for p in target.glob("*")}, before)
                        self.assertEqual(target.exists(), existing)

    def test_no_chunks_does_not_call_api_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            client = response_client(None)
            target = Path(directory) / "index"
            with self.assertRaises(ValueError):
                build_index([document(" \n")], client, target)
            client.embeddings.create.assert_not_called()
            self.assertFalse(target.exists())

    def test_both_strategies_and_legacy_config_work_through_pipeline(self):
        documents = [document("예산 100원.\n\n" * 8, "a"), document("일정 10월.\n\n" * 8, "b")]
        documents[0]["metadata"]["발주 기관"] = "가"
        documents[1]["metadata"]["발주 기관"] = "나"
        for strategy in ("fixed", "boundary"):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as directory, \
                    patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": strategy}):
                client = FakeClient()
                config = build_index(documents, client, directory, "test-embedding", 20, 3)
                self.assertEqual(config["chunking_strategy"], strategy)
                for legacy in (False, True):
                    if legacy:
                        config.pop("chunking_strategy")
                        config.pop("chunking_version")
                        write_json(Path(directory) / "config.json", config)
                    index, chunks, loaded = load_index(directory)
                    self.assertEqual(loaded, config)
                    self.assertEqual(read_json(Path(directory) / "chunks.json"), chunks)
                    hits = retrieve("예산", client, index, chunks, loaded, 1)
                    self.assertEqual(hits[0]["doc_id"], "a")
                    self.assertIsInstance(hits[0]["score"], float)
                    filtered = retrieve("예산", client, index, chunks, loaded, 1, {"발주 기관": "나"})
                    self.assertEqual(filtered[0]["doc_id"], "b")
                    self.assertEqual(retrieve("예산", client, index, chunks, loaded, filters={"발주 기관": "없음"}), [])
                    answer = generate_answer("예산", hits, client)
                    self.assertEqual(answer["sources"][0]["citation"], 1)
                    result = evaluate([{"question": "예산", "expected_doc_ids": ["a"],
                                        "expected_keywords": ["100원"]}], lambda question, filters: answer)
                    self.assertEqual(result["summary"]["recall_at_k"], 1.)
                    self.assertEqual(result["summary"]["keyword_coverage"], 1.)
                    self.assertEqual(json.loads(json.dumps(answer)), answer)
                self.assertEqual(set(client.models), {"test-embedding"})


class IndexIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RFP_CHUNKING_STRATEGY": "fixed"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.documents = [document("예산 100원.\n\n" * 8, "a"), document("일정 10월.\n\n" * 8, "b")]

    def build(self, directory, model="test-embedding", client=None):
        return build_index(self.documents, client or FakeClient(), directory, model, 20, 3)

    def test_saved_files_match_chunking_config_and_vector_order(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.build(directory)
            expected_chunks = chunk_documents(self.documents, 20, 3)
            index, chunks, loaded = load_index(directory)
            self.assertEqual(chunks, expected_chunks)
            self.assertEqual(config, {
                "embedding_model": "test-embedding", "dimension": 2,
                "chunk_count": len(expected_chunks), "chunk_size": 20, "chunk_overlap": 3,
                "chunking_strategy": "fixed", "chunking_version": 1,
            })
            self.assertEqual(loaded, config)
            # FAISS 벡터 i번이 chunks i번 본문의 임베딩이어야 합니다.
            expected = [[1., 0.] if "예산" in c["text"] else [0., 1.] for c in chunks]
            np.testing.assert_allclose(index.reconstruct_n(0, index.ntotal), expected, atol=1e-7)
            raw_chunks = (Path(directory) / "chunks.json").read_text(encoding="utf-8")
            self.assertIn("예산", raw_chunks)
            self.assertNotIn("\\u", raw_chunks)

    def test_batch_boundaries_keep_all_chunks_in_order(self):
        for count, sizes in ((1, [1]), (31, [31]), (32, [32]), (33, [32, 1]),
                             (64, [32, 32]), (65, [32, 32, 1])):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                client = response_client(None)
                client.embeddings.create.side_effect = lambda model, input: response([[1., 1.]] * len(input))
                documents = [document(f"본문 {i}", str(i)) for i in range(count)]
                config = build_index(documents, client, directory, "test-model")
                index, chunks, _ = load_index(directory)
                calls = [c.kwargs["input"] for c in client.embeddings.create.call_args_list]
                self.assertEqual([len(batch) for batch in calls], sizes)
                self.assertEqual([text for batch in calls for text in batch], [c["text"] for c in chunks])
                self.assertEqual((index.ntotal, config["chunk_count"]), (count, count))

    def test_invalid_settings_fail_before_api_or_write(self):
        for env, size, overlap in (({}, 5, 5), ({}, 0, 0), ({"RFP_CHUNKING_STRATEGY": "unknown"}, 20, 3)):
            with self.subTest(env=env, size=size, overlap=overlap), \
                    patch.dict(os.environ, env), tempfile.TemporaryDirectory() as directory:
                client = response_client(None)
                target = Path(directory) / "index"
                with self.assertRaises(ValueError):
                    build_index(self.documents, client, target, "test-model", size, overlap)
                client.embeddings.create.assert_not_called()
                self.assertFalse(target.exists())

    def test_load_rejects_mismatched_saved_files_without_changing_them(self):
        def drop_chunk(chunks, config):
            chunks.pop()

        def drop_chunk_and_count(chunks, config):
            chunks.pop()
            config["chunk_count"] -= 1

        def add_chunk_and_count(chunks, config):
            chunks.append(copy.deepcopy(chunks[0]))
            config["chunk_count"] += 1

        def wrong_dimension(chunks, config):
            config["dimension"] += 1

        def wrong_count(chunks, config):
            config["chunk_count"] += 1

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            self.build(source)
            for tamper in (drop_chunk, drop_chunk_and_count, add_chunk_and_count,
                           wrong_dimension, wrong_count):
                with self.subTest(tamper=tamper.__name__):
                    target = Path(directory) / tamper.__name__
                    shutil.copytree(source, target)
                    chunks, config = read_json(target / "chunks.json"), read_json(target / "config.json")
                    tamper(chunks, config)
                    write_json(target / "chunks.json", chunks)
                    write_json(target / "config.json", config)
                    before = {p.name: p.read_bytes() for p in target.iterdir()}
                    with self.assertRaisesRegex(ValueError, "불일치"):
                        load_index(target)
                    self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, before)

    def test_load_fails_when_any_saved_file_is_missing(self):
        for name in ("index.faiss", "chunks.json", "config.json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                self.build(directory)
                (Path(directory) / name).unlink()
                with self.assertRaises((RuntimeError, OSError)):
                    load_index(directory)

    def test_retrieve_embeds_question_with_saved_model_and_checks_dimension(self):
        with tempfile.TemporaryDirectory() as directory:
            self.build(directory, model="saved-model")
            index, chunks, config = load_index(directory)
            client = FakeClient()
            with patch.dict(os.environ, {"OPENAI_EMBEDDING_MODEL": "other-model"}):
                hits = retrieve("예산", client, index, chunks, config, 1)
            self.assertEqual(client.models, ["saved-model"])
            self.assertEqual(hits[0]["doc_id"], "a")
            wrong = response_client(response([[1., 0., 0.]]))
            with self.assertRaisesRegex(ValueError, "차원"):
                retrieve("예산", wrong, index, chunks, config, 1)


if __name__ == "__main__":
    unittest.main()
