import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import a_embedding as emb
from test_a_overview_chunking import document
from test_a_chunking_experiment import client
from experiments.a_overview_additive_chy import validate_additive, build_reusing_fixed, chunk_key
from experiments import a_chunking_chy as base

class AdditiveTests(unittest.TestCase):
    def setUp(self):
        model = patch.object(base, "MODEL_NAME", "emb")
        model.start()
        self.addCleanup(model.stop)

    def test_fixed_preserved_and_only_extra_overviews_embedded(self):
        docs=[document('Ⅰ 사업 개요\n사업명: 테스트\n사업기간: 100일\nⅡ 요구\n'+'요구 내용. '*400)]
        with tempfile.TemporaryDirectory() as tmp:
            local=client()
            with patch.dict(os.environ,{'RFP_CHUNKING_STRATEGY':'fixed','RFP_EMBEDDING_CONTEXT':'none'}):
                emb.build_index(docs,local,Path(tmp)/'fixed',base.MODEL_NAME,documents_sha256='hash')
            index,chunks,cfg=emb.load_index(Path(tmp)/'fixed')
            fixed=dict(index=index,chunks=chunks,config=cfg)
            original=emb._document_vectors
            with patch.dict(os.environ,{'RFP_CHUNKING_STRATEGY':'overview_additive','RFP_EMBEDDING_CONTEXT':'none'}):
                with patch.object(emb,'_document_vectors',wraps=original) as measured:
                    result,reuse=build_reusing_fixed(docs,local,Path(tmp)/'candidate',fixed,'hash')
                self.assertEqual(sum(len(call.args[0]) for call in measured.call_args_list),1)
                actual_index,actual,_=emb.load_index(Path(tmp)/'candidate')
                self.assertEqual(actual,emb.chunk_documents(docs))
            self.assertEqual(result['chunking_strategy'],'overview_additive')
            self.assertEqual(reuse['newly_embedded'],1)
            self.assertEqual(validate_additive(docs,chunks,actual)['fixed_preserved'],len(chunks))
            old={chunk_key(c):v for c,v in zip(chunks,index.reconstruct_n(0,index.ntotal))}
            for c,v in zip(actual,actual_index.reconstruct_n(0,actual_index.ntotal)):
                if chunk_key(c) in old:np.testing.assert_array_equal(v,old[chunk_key(c)])

    def test_exact_duplicate_and_no_overview(self):
        for doc in [document('Ⅰ 사업 개요\n사업명: 테스트\n사업기간: 100일\n'),document('본문 '*600,[])]:
            with patch.dict(os.environ,{'RFP_CHUNKING_STRATEGY':'fixed'}):fixed=emb.chunk_documents([doc])
            with patch.dict(os.environ,{'RFP_CHUNKING_STRATEGY':'overview_additive'}):actual=emb.chunk_documents([doc])
            self.assertEqual(actual,fixed)
            self.assertEqual(validate_additive([doc],fixed,actual)['added'],0)
