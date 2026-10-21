"""Default strategy and persisted-index compatibility without models or environment files."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipelines.server.server_embedding as emb
from pipelines.api import api_embedding
from pipelines.server.server_generation import generate_answer
from pipelines.server.server_retrieval import retrieve
from test_a_overview_chunking import document
from test_a_chunking_experiment import client


class AdditiveDefaultTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {'RFP_EMBEDDING_CONTEXT': 'none'})
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop('RFP_CHUNKING_STRATEGY', None)

    def sample(self):
        return document('Ⅰ 사업 개요\n사업명: 테스트\n사업기간: 100일\n'
                        'Ⅱ 요구\n' + '요구 내용. ' * 400)

    def test_unset_default_adds_overview_and_preserves_fixed_content(self):
        doc = self.sample()
        before = copy.deepcopy(doc)
        actual = emb.chunk_documents([doc])
        with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': 'overview_additive'}):
            self.assertEqual(actual, emb.chunk_documents([doc]))
        with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': 'fixed'}):
            fixed = emb.chunk_documents([doc])
        self.assertEqual(len(actual), len(fixed) + 1)
        def contents(c):
            return c['doc_id'], c['text'], json.dumps(c['metadata'], sort_keys=True)
        self.assertTrue(set(map(contents, fixed)) <= set(map(contents, actual)))
        self.assertEqual(doc, before)
        self.assertEqual(json.loads(json.dumps(actual)), actual)
        self.assertEqual(len({c['chunk_id'] for c in actual}), len(actual))
        for c in actual:
            self.assertEqual(c['text'], doc['text'][c['metadata']['start_char']:c['metadata']['end_char']])

    def test_no_eligible_overview_and_exact_duplicate_keep_fixed(self):
        docs = [document('본문 ' * 600, []),
                document('Ⅰ 사업 개요\n사업명: 테스트\n' + '내용' * 900),
                document('Ⅰ 사업 개요\n사업명: 테스트\n내용만 있음\nⅡ 요구\n끝'),
                document('Ⅰ 사업 개요\n사업명: 테스트\n사업기간: 100일\n'),
                document(''), document(' \r\n ')]
        for doc in docs:
            with self.subTest(text_length=len(doc['text'])):
                actual = emb.chunk_documents([doc])
                with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': 'fixed'}):
                    self.assertEqual(actual, emb.chunk_documents([doc]))

    def test_table_boundary_inside_table_is_not_added(self):
        text = 'Ⅰ 사업 개요\n사업명: 테스트\n<!-- table:T1 -->\n| 사업기간 | 100일 |\n| 사업예산 | 100원 |\n<!-- /table:T1 -->\nⅡ 요구\n끝'
        doc = document(text, [[text.index('| 사업기간'), '사업 개요'], [text.index('Ⅱ 요구'), '요구']])
        actual = emb.chunk_documents([doc])
        with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': 'fixed'}):
            self.assertEqual(actual, emb.chunk_documents([doc]))

    def test_small_chunks_high_overlap_and_crlf_preserve_source(self):
        doc = document('Ⅰ 사업 개요\r\n사업명: 가\r\n사업기간: 5일\r\nⅡ 요구\r\n' + '가' * 100)
        for size in (1, 40, 80):
            for overlap in (0, size - 1):
                with self.subTest(size=size, overlap=overlap):
                    chunks = emb.chunk_documents([doc], size, overlap)
                    covered = set()
                    for chunk in chunks:
                        a, b = chunk['metadata']['start_char'], chunk['metadata']['end_char']
                        self.assertEqual(chunk['text'], doc['text'][a:b])
                        self.assertLessEqual(b - a, size)
                        covered.update(range(a, b))
                    self.assertTrue(all(i in covered for i, ch in enumerate(doc['text']) if not ch.isspace()))
                    self.assertEqual(len(chunks), len({c['chunk_id'] for c in chunks}))

    def test_invalid_settings_and_empty_build_still_fail(self):
        for value in ('', 'UNKNOWN', 'OVERVIEW_ADDITIVE'):
            with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': value}), self.assertRaises(ValueError):
                emb.chunk_documents([self.sample()])
        for size, overlap in ((0, 0), (10, -1), (10, 10)):
            with self.assertRaises(ValueError):
                emb.chunk_documents([self.sample()], size, overlap)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            emb.build_index([], client(), tmp)

    def test_default_build_load_retrieve_answer_and_old_fixed_load(self):
        local = client()
        with tempfile.TemporaryDirectory() as tmp:
            paths = {name: Path(tmp) / name for name in ('fixed', 'additive')}
            with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': 'fixed'}):
                old = emb.build_index([self.sample()], local, paths['fixed'], 'emb')
            cfg = emb.build_index([self.sample()], local, paths['additive'], 'emb')
            self.assertEqual(cfg['chunking_strategy'], 'overview_additive')
            self.assertEqual(cfg['chunking_version'], 1)
            self.assertEqual({p.name for p in paths['additive'].iterdir()}, {'index.faiss', 'chunks.json', 'config.json'})
            for name in paths:
                index, chunks, saved = emb.load_index(paths[name])
                self.assertEqual(saved, old if name == 'fixed' else cfg)
                self.assertEqual(index.ntotal, len(chunks))
                hits = retrieve('사업기간', local, index, chunks, saved, 5,
                                rerank='none', hybrid=False, rewrite='off', within_doc='off',
                                expand=0, dedup=False, min_score='none', max_per_doc='none')
                answer = generate_answer('사업기간', hits, local, 'test')
                self.assertTrue(hits)
                self.assertEqual(len(answer['sources']), len(hits))
                self.assertEqual([s['citation'] for s in answer['sources']], list(range(1,len(hits)+1)))

    def test_b_default_stays_structured(self):
        self.assertEqual(api_embedding._chunking_strategy(), 'structured')
