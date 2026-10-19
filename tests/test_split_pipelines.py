"""독립 import 경계, 경로 및 두 파이프라인의 실제 FAISS 저장/검색 계약."""
import ast
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_pipeline import FakeClient

ROOT = Path(__file__).resolve().parents[1]


class LocalFake(FakeClient):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def embed_texts(self, texts, model):
        return [[1., 0.] if "예산" in text else [0., 1.] for text in texts]

    def generate_answer(self, question, hits, model):
        import a_generation
        sources = [{"citation": i, **hit} for i, hit in enumerate(hits, 1)]
        return a_generation._finish(question, sources, model, "예산은 100원입니다. [1]", False)


class SplitPipelineTests(unittest.TestCase):
    def test_project_dependencies_stay_inside_each_scenario(self):
        project = {p.stem for p in ROOT.glob('*.py')}
        for prefix in ('a_', 'b_'):
            for path in ROOT.glob(prefix + '*.py'):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    names = ([node.module] if isinstance(node, ast.ImportFrom) else
                             [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                    # Also inspect dynamic import module names.
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'load_component', 'import_module'}:
                        if node.args and isinstance(node.args[0], ast.Constant):
                            names.append(node.args[0].value)
                    for name in names:
                        if name in project:
                            self.assertTrue(name.startswith(prefix), (path.name, name))

    def test_fixed_scenario_and_separate_default_paths(self):
        for prefix in ('a', 'b'):
            module = importlib.import_module(prefix + '_run')
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ('.env.' + prefix)).write_text('A_GENERATION_MODEL=gen\nA_EMBEDDING_MODEL=emb\n')
                with patch.dict(os.environ, {'RFP_SCENARIO': 'B' if prefix == 'a' else 'A'}, clear=True), \
                        patch.object(module, 'ROOT', root), patch('sys.argv', [prefix+'_run.py', 'build', '--owner', 'test']), \
                        patch.object(module, 'run_pipeline') as pipeline, \
                        patch.object(module, 'next_experiment_id', return_value='test-0001') as ids, \
                        redirect_stdout(io.StringIO()):
                    module.main()
                args = pipeline.call_args.args[0]
                self.assertEqual(args.scenario_settings['scenario'], prefix.upper())
                self.assertEqual(args.processed_dir, root / 'data/processed' / prefix)
                self.assertEqual(args.index_dir, root / 'indexes' / prefix)
                self.assertEqual(args.results_dir, root / 'results' / prefix)
                self.assertEqual(args.reports_dir, root / 'results' / prefix / 'reports')
                self.assertEqual(ids.call_args.args[1], root / '.experiment-state' / prefix)

    def test_both_build_search_generate_and_isolate_trace_state(self):
        import a_client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix in ('a', 'b'):
                module = importlib.import_module(prefix + '_run')
                obs = importlib.import_module(prefix + '_observability')
                other = importlib.import_module(('b' if prefix == 'a' else 'a') + '_observability')
                args = SimpleNamespace(command='build', processed_dir=root/prefix/'processed',
                    index_dir=root/prefix/'index', results_dir=root/prefix/'results', owner='test',
                    chunk_size=1000, chunk_overlap=150, top_k=1, question='예산?',
                    scenario_settings={'scenario':prefix.upper(), 'A_GENERATION_MODEL':'gen', 'A_EMBEDDING_MODEL':'emb'})
                module.write_json(args.processed_dir/'documents.json', [{'doc_id':'one', 'text':'예산은 100원입니다.', 'metadata':{'filename':'one.txt'}}])
                trace = obs.Trace({})
                with patch.dict(os.environ, {'OPENAI_API_KEY':'fake', 'OPENAI_GENERATION_MODEL':'gpt-5-mini',
                        'RETRIEVAL_REWRITE':'off', 'RETRIEVAL_RERANK':'none', 'RFP_CHUNKING_STRATEGY':'fixed', 'RFP_EMBEDDING_CONTEXT':'none'}), \
                        patch.object(module, 'OpenAI', return_value=FakeClient()), \
                        patch.object(a_client, 'ScenarioAClient', LocalFake), \
                        redirect_stdout(io.StringIO()), trace.run('split-test'):
                    self.assertIsNone(other._active.get())
                    module.run_pipeline(args, __import__('argparse').ArgumentParser(), {}, {'experiment_id':'test-0001'}, trace, 'test')
                    args.command='ask'
                    module.run_pipeline(args, __import__('argparse').ArgumentParser(), {}, {'experiment_id':'test-0001'}, trace, 'test')
                result=json.loads((args.results_dir/'ask_test-0001_test.json').read_text())
                self.assertEqual(result['sources'][0]['doc_id'], 'one')
            self.assertTrue((root/'a/index/index.faiss').is_file())
            self.assertTrue((root/'b/index/index.faiss').is_file())

    def test_dispatch_imports_only_selected_entrypoint(self):
        import run
        for prefix in ('a', 'b'):
            module=importlib.import_module(prefix+'_run')
            with patch.dict(os.environ, {'RFP_SCENARIO':prefix.upper()}), patch.object(module, 'main') as main:
                run.main()
            main.assert_called_once_with()

    def test_retrieval_scenario_does_not_follow_other_environment(self):
        import a_retrieval
        import b_retrieval
        with patch.dict(os.environ, {'RFP_SCENARIO': 'B', 'A_RERANK_MODEL': ''}, clear=True), \
                patch.object(a_retrieval, '_cross_encoder_available', return_value=True):
            self.assertEqual(a_retrieval._min_score(None), a_retrieval.DEFAULT_MIN_SCORE_A)
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(a_retrieval.retrieval_options()['rerank'], 'lexical')
        with patch.dict(os.environ, {'RFP_SCENARIO': 'A'}, clear=True):
            self.assertEqual(b_retrieval._min_score(None), b_retrieval.DEFAULT_MIN_SCORE)
            self.assertEqual(b_retrieval.rerank_model_source(), b_retrieval.DEFAULT_RERANK_MODEL)

    def test_evaluation_uses_its_own_trace(self):
        for prefix in ('a', 'b'):
            evaluation=importlib.import_module(prefix+'_evaluation')
            obs=importlib.import_module(prefix+'_observability')
            trace=obs.Trace({})
            with trace.run('evaluation-isolation'):
                self.assertEqual(evaluation._usage_snapshot(), {})
                result=evaluation.evaluate([{'question':'예산?', 'expected_doc_ids':['one'], 'expected_keywords':['100원']}],
                    lambda question, filters: {'answer':'100원', 'sources':[], 'status':'ok'})
            self.assertEqual(result['summary']['keyword_coverage'], 1.0)
