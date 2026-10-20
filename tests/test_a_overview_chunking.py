import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pipelines.server.server_embedding as emb
from parsing.parsing_documents import _sections
from pipelines.server.server_retrieval import retrieve
from pipelines.server.server_generation import generate_answer
from evaluation.evaluation_metrics import evaluate
from experiments import a_chunking_chy as base
from test_a_chunking_experiment import client


def document(text,sections=None):
    return {'doc_id':'d','text':text,'metadata':{'filename':'test','사업명':'테스트'},
            'sections':_sections(text) if sections is None else sections}


class OverviewTests(unittest.TestCase):
    def setUp(self):
        env=patch.dict(os.environ,{'RFP_CHUNKING_STRATEGY':'overview','RFP_EMBEDDING_CONTEXT':'none'})
        env.start();self.addCleanup(env.stop)

    def test_overview_is_complete_and_other_text_covered(self):
        before='Ⅰ 배경\n'+'주변 내용. '*180+'\n'
        block='Ⅱ 사업 개요\n사업명: 테스트\n사업기간: 100일\n소요예산: 100원\n'
        after='Ⅲ 요구사항\n'+'다른 내용. '*180
        doc=document(before+block+after);snapshot=copy.deepcopy(doc)
        chunks=emb.chunk_documents([doc])
        self.assertIn(block,[c['text'] for c in chunks])
        base.geometry([doc],chunks)
        self.assertEqual(doc,snapshot)
        self.assertEqual(json.loads(json.dumps(chunks)),chunks)

    def test_no_sections_long_or_single_field_falls_back_exactly_to_fixed(self):
        texts=[document('본문'*900,[]),document('Ⅰ 사업 개요\n사업명: 테스트\n'+'내용'*900),
               document('Ⅰ 사업 개요\n사업명: 테스트\n내용만 있음\nⅡ 요구\n다음')]
        for doc in texts:
            candidate=emb.chunk_documents([doc])
            with patch.dict(os.environ,{'RFP_CHUNKING_STRATEGY':'fixed'}):
                self.assertEqual(candidate,emb.chunk_documents([doc]))

    def test_parent_title_does_not_select_child(self):
        text='추진배경\n사업기간: 100일\n사업예산: 100원\n'
        doc=document(text,[[0,'Ⅰ 사업 개요 > 1 추진배경']])
        self.assertEqual(emb._overview_spans(doc,1000,[]),[])

    def test_short_table_preserved_and_boundary_inside_table_rejected(self):
        text='Ⅰ 사업 개요\n사업명: 테스트\n<!-- table:T1 -->\n| 사업기간 | 100일 |\n| 사업예산 | 100원 |\n<!-- /table:T1 -->\nⅡ 요구\n끝'
        doc=document(text)
        chunks=emb.chunk_documents([doc]);self.assertIn('<!-- /table:T1 -->',chunks[0]['text'])
        start=text.index('| 사업기간')
        doc['sections']=[[start,'사업 개요'],[text.index('Ⅱ 요구'),'요구']]
        self.assertEqual(emb._overview_spans(doc,1000,emb._table_spans(text)),[])

    def test_tiny_sizes_high_overlap_crlf_empty_and_malformed_sections(self):
        for text in ['', ' \r\n ', 'Ⅰ 사업 개요\r\n사업명: 가\r\n사업기간: 5일\r\nⅡ 요구\r\n'+'가'*100]:
            doc=document(text)
            doc['sections'] += [[False,'사업 개요'],[-1,'사업 개요'],[2,None]]
            for size in [1,10,40,1000]:
                for overlap in [0,size-1]:
                    chunks=emb.chunk_documents([doc],size,overlap)
                    covered=set()
                    for c in chunks:
                        a,b=c['metadata']['start_char'],c['metadata']['end_char']
                        self.assertEqual(c['text'],text[a:b]);self.assertLessEqual(b-a,size)
                        covered.update(range(a,b))
                    self.assertTrue(all(i in covered for i,ch in enumerate(text) if not ch.isspace()))
                    self.assertEqual(len({c['chunk_id'] for c in chunks}),len(chunks))

    def test_build_load_retrieve_answer_contract(self):
        doc=document('Ⅰ 사업 개요\n사업명: 테스트\n사업기간: 100일\n사업예산: 100원\nⅡ 요구\n다음 내용')
        local=client()
        with tempfile.TemporaryDirectory() as tmp:
            cfg=emb.build_index([doc],local,tmp,'emb',documents_sha256='hash')
            index,chunks,saved=emb.load_index(tmp)
            self.assertEqual(cfg,saved);self.assertEqual(cfg['chunking_strategy'],'overview')
            self.assertEqual(cfg['chunking_version'],1)
            hits=retrieve('기간',local,index,chunks,cfg,5,**base.PROFILES['dense'])
            answer=generate_answer('기간',hits,local,'gen')
            result=evaluate([{'question':'기간','expected_doc_ids':['d']}],lambda q,f:answer)
            self.assertEqual(result['summary']['recall_at_k'],1)
            self.assertEqual({p.name for p in Path(tmp).iterdir()},{'index.faiss','chunks.json','config.json'})

    def test_malformed_strategy_still_rejected(self):
        with patch.dict(os.environ,{'RFP_CHUNKING_STRATEGY':'wrong'}):
            with self.assertRaises(ValueError):emb.chunk_documents([])


class OverviewEvaluationTests(unittest.TestCase):
    def test_strict_evidence_requires_whole_labeled_span_and_correct_document(self):
        from experiments.a_overview_chy import strict_evidence
        case={'evidence_spans':[{'doc_id':'a','start_char':10,'end_char':30}]}
        hit={'doc_id':'a','metadata':{'start_char':10,'end_char':30}}
        self.assertEqual(strict_evidence(case,[hit]),1)
        self.assertEqual(strict_evidence(case,[{**hit,'doc_id':'b'}]),0)
        self.assertEqual(strict_evidence(case,[{'doc_id':'a','metadata':{'start_char':10,'end_char':29}}]),0)
        self.assertIsNone(strict_evidence({},[hit]))

    def test_rrf_capture_keeps_original_rankings_and_results(self):
        from experiments.a_overview_chy import capture
        import pipelines.server.server_retrieval as retrieval
        chunks=[{'chunk_id':str(i)} for i in range(4)]
        ranks=[[2,0,1],[1,3,0]]
        expected=retrieval._rrf(ranks,60)
        original=retrieval._rrf
        with capture(chunks) as stages:
            self.assertEqual(retrieval._rrf(ranks,60),expected)
            self.assertEqual(stages['vector_ids'],['2','0','1'])
            self.assertEqual(stages['bm25_ids'],['1','3','0'])
        self.assertIs(retrieval._rrf,original)
        self.assertEqual(ranks,[[2,0,1],[1,3,0]])
