"""A 청킹 실험의 통제 조건, 실제 A 계약, 지표 분모 및 재사용 보호."""
import copy
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import a_embedding
from a_client import ScenarioAClient
from a_generation import generate_answer
from a_observability import Trace
from a_parsing import _sections, write_json
from a_retrieval import retrieve, retrieval_options
from experiments import a_chunking_chy as experiment
from test_observability import FakeLangfuse


def document(text):
    return {'doc_id':'d','text':text,'metadata':{'filename':'PRIVATE_FILE','사업명':'PRIVATE_PROJECT'},
            'sections':_sections(text)}


def client():
    obj=ScenarioAClient.__new__(ScenarioAClient)
    obj.embedding_model='emb'
    obj.embedding=SimpleNamespace(embed_texts=lambda texts:np.array([[len(t)+1,1] for t in texts]))
    obj.generation=SimpleNamespace(generate_answer=lambda q,h,m:{'question':q,'answer':'answer [1]',
        'model':m,'status':'ok','sources':[{'citation':i,**hit} for i,hit in enumerate(h,1)]})
    return obj


class AChunkingContractTests(unittest.TestCase):
    def test_public_signatures_remain_unchanged(self):
        expected={a_embedding.chunk_documents:'(documents, chunk_size=1000, chunk_overlap=150)',
                  a_embedding.embed_texts:'(texts, client, model)',
                  a_embedding.build_index:"(documents, client, index_dir='indexes/a', model='text-embedding-3-small', chunk_size=1000, chunk_overlap=150, documents_sha256=None)",
                  a_embedding.load_index:"(index_dir='indexes/a')"}
        for fn,signature in expected.items(): self.assertEqual(str(inspect.signature(fn)),signature)

    def test_four_variants_coordinates_tables_json_and_full_connection(self):
        table='<!-- table:T1 -->\n| 항목 | 값 |\n| --- | --- |\n| 예산 | 100원 |\n<!-- /table:T1 -->'
        doc=document('목차\nⅠ 개요\n'+'소개 문장. '*120+'\n'+table+'\nⅡ 요구\n'+'요구 문장. '*150)
        before=copy.deepcopy(doc)
        for variant in experiment.VARIANTS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{},clear=True):
                experiment.configure(variant)
                fake=FakeLangfuse(); trace=Trace({},fake); local=client()
                with trace.run('contract'):
                    config=a_embedding.build_index([doc],local,directory,'emb',documents_sha256='source-hash')
                    index,chunks,loaded=a_embedding.load_index(directory)
                    geometry=experiment.geometry([doc],chunks)
                    hits=retrieve('PRIVATE_QUESTION',local,index,chunks,loaded,5,**experiment.PROFILES['dense'])
                    answer=generate_answer('PRIVATE_QUESTION',hits,local,'gen')
                self.assertEqual(config,loaded)
                self.assertEqual(geometry['chunk_count'],index.ntotal)
                self.assertEqual(config['documents_sha256'],'source-hash')
                self.assertEqual(answer['sources'][0]['doc_id'],'d')
                self.assertEqual(answer['sources'][0]['citation'],1)
                self.assertEqual(json.loads(json.dumps(answer)),answer)
                vectors=index.reconstruct_n(0,index.ntotal)
                self.assertEqual(vectors.dtype,np.float32)
                np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-6)
                if variant.startswith('structured'):
                    self.assertEqual(geometry['small_tables_not_whole'],0)
                self.assertNotIn('PRIVATE_',str(fake.records))
        self.assertEqual(doc,before)

    def test_boundaries_empty_crlf_long_rows_malformed_sections_high_overlap(self):
        texts=['',' \r\n ', '가'*1001,
               '목 차\r\nⅠ 개요\r\n'+'내용\r\n'*60,
               '<!-- table:T1 -->\n|'+'값'*1200+'|\n<!-- /table:T1 -->']
        for variant in experiment.VARIANTS:
            for text in texts:
                with self.subTest(variant=variant,length=len(text)),patch.dict(os.environ,{},clear=True):
                    experiment.configure(variant)
                    doc=document(text);doc['sections']=[[False,'bad'],[-1,'bad'],[2,None],[0,'개요']]
                    original=copy.deepcopy(doc)
                    chunks=a_embedding.chunk_documents([doc],40,39)
                    covered=set()
                    for chunk in chunks:
                        start,end=chunk['metadata']['start_char'],chunk['metadata']['end_char']
                        self.assertEqual(chunk['text'],text[start:end]);self.assertLessEqual(end-start,40)
                        covered.update(range(start,end))
                    self.assertTrue(all(i in covered for i,c in enumerate(text) if not c.isspace()))
                    self.assertEqual(doc,original)
        for size,overlap in ((0,0),(10,10),(10,-1)):
            with self.assertRaises(ValueError): a_embedding.chunk_documents([],size,overlap)

    def test_failed_vectors_preserve_index_and_model_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True):
            experiment.configure('fixed');obj=client();docs=[document('본문')]
            a_embedding.build_index(docs,obj,d,'emb')
            before={p.name:p.read_bytes() for p in Path(d).iterdir()}
            for vectors in ([[0,0]],[[float('nan'),1]],[[float('inf'),1]],[[]],[[1,2],[3,4]]):
                with patch.object(obj.embedding,'embed_texts',return_value=vectors):
                    with self.assertRaises(ValueError): a_embedding.build_index(docs,obj,d,'emb')
                self.assertEqual(before,{p.name:p.read_bytes() for p in Path(d).iterdir()})
            with self.assertRaises(ValueError): a_embedding.embed_texts(['q'],obj,'different')


class ExperimentTests(unittest.TestCase):
    def test_gpu_wait_is_bounded_and_does_not_start_another_model(self):
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,
            mem_get_info=Mock(side_effect=[(1,10*1024**3),(9*1024**3,10*1024**3)])))
        with patch.object(experiment.time,'sleep') as sleep,patch('sys.stdout',new=io.StringIO()):
            experiment.wait_for_gpu(torch,'a-default',30)
            sleep.assert_called_once()
        torch.cuda.mem_get_info=Mock(return_value=(1,10*1024**3))
        with self.assertRaises(RuntimeError):experiment.wait_for_gpu(torch,'dense',0)

    def test_explicit_options_override_environment(self):
        hostile={'RETRIEVAL_REWRITE':'both','RETRIEVAL_DEDUP':'true','RETRIEVAL_MIN_SCORE':'0.99',
                 'RETRIEVAL_WITHIN_DOC':'residual','RETRIEVAL_RERANK':'lexical','RETRIEVAL_HYBRID':'false',
                 'RFP_EMBEDDING_CONTEXT':'section','RFP_CHUNKING_STRATEGY':'structured'}
        with patch.dict(os.environ,hostile):
            experiment.configure('fixed')
            self.assertEqual(os.environ['RFP_EMBEDDING_CONTEXT'],'none')
            for name in experiment.PROFILES:
                opts=retrieval_options(**experiment.PROFILES[name])
                self.assertFalse(opts['dedup']);self.assertIsNone(opts['min_score'])
                self.assertEqual(opts['rewrite'],'off');self.assertEqual(opts['within_doc'],'off')
                self.assertEqual(opts['hybrid'],name=='a-default')
            self.assertEqual(retrieval_options(**experiment.PROFILES['a-default'])['candidates'],50)

    def test_denominators_negative_cases_and_failures(self):
        case={'id':'yes','question':'private','expected_doc_ids':['d'],'expected_keywords':['100원']}
        hit={'doc_id':'d','chunk_id':'d:0','text':'100원','score':.9}
        rows=[]
        for repetition in range(3):
            rows += [experiment.case_record(case,[hit],repetition,.1),
                     experiment.case_record(dict(case,id='failure'),[],repetition,.2,'RuntimeError'),
                     experiment.case_record({'id':'no','answerable':False,'expected_doc_ids':['d']},[hit],repetition,.3)]
        result=experiment.summarize(rows)
        self.assertEqual(result['recall_at_5'],.5);self.assertEqual(result['mrr_at_5'],.5)
        self.assertEqual(result['answer_chunk_hit_rate'],.5)
        self.assertEqual(result['answerable_cases'],2);self.assertEqual(result['unanswerable_cases'],1)
        self.assertEqual(result['failures'],3);self.assertEqual(result['negative_empty_rate'],0)
        self.assertNotIn('private',str(rows));self.assertNotIn('text',rows[0]['hits'][0])

    def test_validation_allows_negative_no_gold_but_rejects_missing_positive(self):
        docs=[document('body')]
        good=[{'id':'p','question':'q','expected_doc_ids':['d']},
              {'id':'n','question':'q','answerable':False,'expected_doc_ids':[]}]
        experiment.validate_inputs(docs,good)
        for bad in ([dict(good[0],expected_doc_ids=[])],[dict(good[0],expected_doc_ids=['missing'])],
                    good+[good[0]],[dict(good[0],question='')],good[1:]):
            with self.assertRaises(ValueError):experiment.validate_inputs(docs,bad)
        with self.assertRaises(ValueError):experiment.validate_inputs(docs+docs,good)

    def test_index_reuse_checks_input_model_code_and_files(self):
        with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{},clear=True):
            path=Path(d);experiment.configure('fixed')
            a_embedding.build_index([document('body')],client(),path,'emb',documents_sha256='documents')
            manifest=dict(documents_sha256='documents',evaluation_sha256='evaluation',
                          model_manifest_sha256='model',code_hashes={'a':'code'},index_hashes=experiment.fingerprints(path))
            args=(path,manifest,'documents','evaluation','model',{'a':'code'})
            experiment.validate_reuse(*args)
            for key in ('documents_sha256','evaluation_sha256','model_manifest_sha256','code_hashes'):
                with self.assertRaises(ValueError):
                    experiment.validate_reuse(path,{**manifest,key:'changed'},*args[2:])
            (path/'chunks.json').write_text('[]')
            with self.assertRaises(ValueError):experiment.validate_reuse(*args)

    def test_selection_follows_recall_chunk_hit_mrr_then_size_and_refuses_failures(self):
        def result(recall=1,hit=.5,mrr=.8,size=10,failures=0):
            return {'summary':dict(recall_at_5=recall,answer_chunk_hit_rate=hit,mrr_at_5=mrr,failures=failures),
                    'config':{'chunk_count':size}}
        base=result()
        for better in (result(hit=.6),result(mrr=.9),result(size=9)):
            self.assertGreater(experiment.rank_key(better),experiment.rank_key(base))
        self.assertGreater(experiment.rank_key(base),experiment.rank_key(result(recall=.9,hit=1)))
        with self.assertRaises(ValueError):experiment.rank_key(result(failures=1))

    def test_report_does_not_include_unapproved_content(self):
        row={'experiment':{'experiment_id':'test-0001','commit':'test','secret':'PRIVATE'},
             'variant':'fixed','profile':'dense','config':{'embedding_model':'emb','chunk_count':1,'dimension':2},
             'summary':{'recall_at_5':1},'timing':{},'stats':{},'records':[{'question':'PRIVATE'}]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'report.md';experiment.report_run(row,path)
            self.assertNotIn('PRIVATE',path.read_text())

    def test_compare_processes_are_isolated_and_reuse_selected_indexes(self):
        results=[]
        def child(command,**kwargs):
            variant=command[command.index('--variant')+1] if '--variant' in command else 'reused'
            idx=len(results)
            row={'variant':variant,'profile':'dense' if idx<4 else 'a-default',
                 'experiment':{'experiment_id':f'test-{idx}','commit':'test'},
                 'summary':dict(recall_at_5=[.5,.6,.8,.7,.9,.95][idx],mrr_at_5=.5,answer_chunk_hit_rate=.5,
                                mean_seconds=.1,p95_seconds=.2,failures=0,answerable_cases=42),
                 'config':{'chunk_count':10},'paths':{'processed':f'p{idx}','index':f'i{idx}','results':str(root/'results')},
                 'rerank_files':{}}
            path=root/f'row{idx}.json';write_json(path,row)
            Path(command[command.index('--result-pointer')+1]).write_text(str(path))
            results.append(command)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'results/reports').mkdir(parents=True)
            args=SimpleNamespace(documents=root/'docs',eval_file=root/'eval',repeat=3,gpu_wait_seconds=600)
            with patch.object(experiment,'ROOT',root),patch.object(experiment,'code_hashes',return_value={}), \
                    patch.object(experiment.subprocess,'run',side_effect=child):
                experiment.compare(args)
            self.assertEqual(len(results),6)
            self.assertIn('p0',results[4]);self.assertIn('i0',results[4])
            self.assertIn('p2',results[5]);self.assertIn('i2',results[5])
            for command in results:self.assertNotIn('--experiment-id',command)


if __name__=='__main__': unittest.main()
