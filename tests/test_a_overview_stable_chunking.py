import os
from unittest.mock import patch

import pipelines.server.server_embedding as emb
from test_a_overview_chunking import OverviewTests, document
from experiments import a_chunking_chy as base


class StableOverviewTests(OverviewTests):
    def setUp(self):
        env = patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': 'overview_stable', 'RFP_EMBEDDING_CONTEXT': 'none'})
        env.start()
        self.addCleanup(env.stop)

    def test_build_load_retrieve_answer_contract(self):
        # Parent contract pins the original strategy name; separately exercise
        # candidate persistence through the existing fake embedding client.
        import tempfile
        from test_a_chunking_experiment import client
        doc = document('Ⅰ 사업 개요\n사업명: 테스트\n사업기간: 100일\nⅡ 요구\n' + '요구사항. ' * 400)
        with tempfile.TemporaryDirectory() as path:
            cfg = emb.build_index([doc], client(), path, 'emb', documents_sha256='hash')
            index, chunks, loaded = emb.load_index(path)
            self.assertEqual(cfg, loaded)
            self.assertEqual(cfg['chunking_strategy'], 'overview_stable')
            self.assertEqual(chunks, emb.chunk_documents([doc]))
            self.assertEqual(index.ntotal, len(chunks))

    def test_untouched_fixed_chunks_preserved_and_overview_isolated(self):
        before = 'Ⅰ 배경\n' + '배경 내용. ' * 250 + '\n'
        block = 'Ⅱ 사업 개요\n사업명: 테스트\n사업기간: 100일\n사업예산: 100원\n'
        doc = document(before + block + 'Ⅲ 요구사항\n' + '다른 내용. ' * 600)
        stable = emb.chunk_documents([doc])
        with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY': 'fixed'}):
            fixed = emb.chunk_documents([doc])
        spans = emb._overview_spans(doc, 1000, [])
        self.assertTrue(spans)
        coords = lambda c: (c['metadata']['start_char'], c['metadata']['end_char'])
        actual = {coords(c): c for c in stable}
        for c in fixed:
            a, b = coords(c)
            if all(b <= left or a >= right for left, right in spans):
                self.assertIn((a, b), actual)
                self.assertEqual(c['text'], actual[a, b]['text'])
                self.assertEqual(c['metadata'], actual[a, b]['metadata'])
        for a, b in actual:
            for left, right in spans:
                self.assertTrue((a, b) == (left, right) or b <= left or a >= right)
        self.assertEqual(len(actual), len(stable))
        base.geometry([doc], stable)
