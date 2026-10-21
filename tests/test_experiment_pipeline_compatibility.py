"""실험의 현재 모듈 경계와 캐시를 이용한 B 검색 실험의 실제 입출력을 검증합니다."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipelines.server import server_embedding
from pipelines.api import api_embedding
from parsing.parsing_documents import read_json, write_json
from experiments import retrieval_eval_yjk
from test_pipeline import FakeClient

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED = {'embedding', 'generation',
            'query_rewrite', 'experiment_ids', 'experiment_reports', 'scenario_a',
            'scenario_a_embedding', 'scenario_a_generation', 'legacy_run', 'legacy'}


class ExperimentCompatibilityTests(unittest.TestCase):
    def test_active_tests_and_experiments_do_not_import_archived_modules(self):
        paths = [*ROOT.joinpath('tests').glob('*.py'), *ROOT.joinpath('experiments').glob('*.py'), ROOT/'exp_max_per_doc.py']
        for path in paths:
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module] if isinstance(node, ast.ImportFrom) else
                         [item.name for item in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    self.assertNotIn(name, ARCHIVED, (path.name, name))

    def test_b_cached_retrieval_experiment_writes_results_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'b'
            documents = [{'doc_id':'one', 'text':'예산 100원', 'metadata':{'filename':'one.txt'}}]
            with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY':'fixed', 'RFP_EMBEDDING_CONTEXT':'none',
                    'RETRIEVAL_REWRITE':'off', 'RETRIEVAL_RERANK':'none'}, clear=True):
                api_embedding.build_index(documents, FakeClient(), root/'indexes', model='test')
                write_json(root/'eval.json', [{'group':'synthetic','question':'예산?', 'expected_doc_ids':['one']}])
                key = hashlib.sha256('test\n예산?'.encode()).hexdigest()
                write_json(root/'results/query_embedding_cache.json', {key:[1.,0.]})
                with patch.object(retrieval_eval_yjk.CachedEmbeddingClient, 'client', side_effect=AssertionError('API 호출 금지')) as api, redirect_stdout(io.StringIO()):
                    retrieval_eval_yjk.run(root/'indexes', root/'eval.json', root/'results', root/'reports', 'smoke', 1)
            result = read_json(root/'results/retrieval_eval_smoke.json')
            api.assert_not_called()
            self.assertEqual(result['records'][0]['recall_at_k'], 1.0)
            self.assertTrue((root/'reports/retrieval_eval_yjk_smoke.md').is_file())

    def test_scenario_chunking_defaults_are_preserved(self):
        documents = [{'doc_id':'one', 'text':'가나다\n\n라마바'*3, 'metadata':{}}]
        for module, strategy in ((server_embedding,'fixed'),(api_embedding,'boundary')):
            with patch.dict(os.environ, {}, clear=True):
                default = module.chunk_documents(documents, 10, 2)
            with patch.dict(os.environ, {'RFP_CHUNKING_STRATEGY':strategy}, clear=True):
                self.assertEqual(default,module.chunk_documents(documents,10,2))
