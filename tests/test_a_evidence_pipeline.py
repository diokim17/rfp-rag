import unittest
from unittest.mock import patch
from experiments.a_evidence_pipeline_chy import diagnose,capture_stages,OPTIONS
import pipelines.server.server_retrieval as retrieval


def chunk(i,text='target',doc='a'):
    return dict(chunk_id=str(i),doc_id=doc,text=text,metadata={})


class EvidenceStagesTests(unittest.TestCase):
    def setUp(self):
        self.case={'expected_doc_ids':['a'],'expected_keywords':['target']}
        self.gold=chunk(1)
        self.other=chunk(2,'other')
        self.documents=[dict(doc_id='a',text='target',metadata={})]

    def diag(self,chunks=None,captured=None,hits=None):
        return diagnose(self.case,[self.gold] if chunks is None else chunks,self.documents,
                        {} if captured is None else captured,[] if hits is None else hits)

    def test_hit_and_unscored(self):
        self.assertEqual(self.diag(hits=[self.gold])['category'],'hit')
        self.case['expected_keywords']=[]
        self.assertEqual(self.diag()['category'],'not_scored')

    def test_document_cap_requires_uncapped_top5_evidence(self):
        captured={'candidates':[self.gold,self.other],'ranked':[self.other,self.gold]}
        self.assertEqual(self.diag(captured=captured,hits=[self.other])['category'],'document_cap_loss')
        captured['ranked']=[chunk(i,'other') for i in range(10,15)]+[self.gold]
        self.assertEqual(self.diag(captured=captured,hits=[self.other])['category'],'rerank_below_top5')

    def test_missing_candidates_and_threshold(self):
        self.assertEqual(self.diag()['category'],'filter_or_threshold')
        captured={'candidates':[self.other],'ranked':[self.other]}
        self.assertEqual(self.diag(captured=captured)['category'],'candidate_miss')

    def test_annotation_and_chunk_coverage_are_not_asserted_as_chunker_fault(self):
        self.assertEqual(self.diag(chunks=[self.other])['category'],'chunk_coverage_or_annotation')
        self.documents[0]['text']='unrelated'
        self.assertEqual(self.diag(chunks=[self.other])['category'],'annotation_or_source')

    def test_wrong_document_not_evidence(self):
        wrong=chunk(3,doc='b')
        self.assertEqual(self.diag(captured={'candidates':[wrong],'ranked':[wrong]},hits=[wrong])['category'],'candidate_miss')

    def test_capture_delegates_once_and_restores_even_after_error(self):
        original=[self.gold,self.other]
        expanded=original+[chunk(3)]
        ranked=list(reversed(expanded))
        with patch.object(retrieval,'_expand_candidates',return_value=expanded) as expand, \
             patch.object(retrieval,'_rerank',return_value=ranked) as rerank:
            with self.assertRaises(RuntimeError):
                with capture_stages() as stages:
                    self.assertIs(retrieval._expand_candidates('q',original),expanded)
                    self.assertIs(retrieval._rerank('q',expanded),ranked)
                    self.assertEqual(stages['fused'],original)
                    self.assertEqual(stages['ranked'],ranked)
                    raise RuntimeError('test')
            self.assertIs(retrieval._rerank,rerank)
            self.assertIs(retrieval._expand_candidates,expand)
            expand.assert_called_once();rerank.assert_called_once()
        self.assertEqual(original,[self.gold,self.other])

    def test_latest_defaults_pinned(self):
        resolved=retrieval.retrieval_options(**OPTIONS)
        self.assertEqual(resolved['min_score'],retrieval.DEFAULT_MIN_SCORE_A)
        self.assertEqual(resolved['rerank_section'],retrieval.DEFAULT_RERANK_SECTION)
        self.assertEqual(resolved['uncap_named'],retrieval.DEFAULT_UNCAP_NAMED)
        self.assertEqual(resolved['max_per_doc'],retrieval.DEFAULT_MAX_PER_DOC)
