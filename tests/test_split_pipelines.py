"""독립 import 경계, 경로 및 두 파이프라인의 실제 FAISS 저장/검색 계약."""
import ast
from contextlib import redirect_stdout
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
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
        from pipelines.server import server_generation
        sources = [{"citation": i, **hit} for i, hit in enumerate(hits, 1)]
        return server_generation._finish(question, sources, model, "예산은 100원입니다. [1]", False)


class SplitPipelineTests(unittest.TestCase):
    def test_project_dependencies_stay_inside_each_scenario(self):
        shared = ('parsing.', 'evaluation.', 'observability.', 'tracking.', 'retrieval.', 'config.')
        paths = [*ROOT.joinpath('pipelines').rglob('*.py')]
        for folder in ('parsing', 'evaluation', 'observability', 'tracking', 'retrieval', 'config'):
            paths.extend(ROOT.joinpath(folder).glob('*.py'))
        self.assertTrue(paths)
        for path in paths:
            relative = path.relative_to(ROOT)
            allowed = shared
            if relative.parts[0] == 'pipelines':
                allowed += ('pipelines.' + relative.parts[1] + '.',)
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module] if isinstance(node, ast.ImportFrom) else
                         [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'load_component', 'import_module'}:
                    if node.args and isinstance(node.args[0], ast.Constant):
                        names.append(node.args[0].value)
                for name in names:
                    if name and name.startswith(('pipelines.', *shared)):
                        self.assertTrue(any(name == prefix.rstrip('.') or name.startswith(prefix)
                                            for prefix in allowed), (str(relative), name))

    def test_fixed_scenario_and_separate_default_paths(self):
        for prefix in ('a', 'b'):
            module = importlib.import_module('pipelines.server.server_run' if prefix == 'a' else 'pipelines.api.api_run')
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ('.env.' + prefix)).write_text('A_GENERATION_MODEL=gen\nA_EMBEDDING_MODEL=emb\n')
                with patch.dict(os.environ, {'RFP_SCENARIO': 'B' if prefix == 'a' else 'A'}, clear=True), \
                        patch.object(module, 'ROOT', root), patch('sys.argv', ['server_run.py' if prefix == 'a' else 'api_run.py', 'build', '--owner', 'test']), \
                        patch.object(module, 'run_pipeline') as pipeline, \
                        patch.object(module, 'next_experiment_id', return_value='test-0001') as ids, \
                        redirect_stdout(io.StringIO()):
                    module.main()
                args = pipeline.call_args.args[0]
                self.assertEqual(args.scenario_settings['scenario'], prefix.upper())
                self.assertEqual(args.processed_dir, root / 'data/processed/common')
                self.assertEqual(args.index_dir, root / 'indexes' / prefix)
                self.assertEqual(args.results_dir, root / 'results' / prefix)
                self.assertEqual(args.reports_dir, root / 'results' / prefix / 'reports')
                self.assertEqual(ids.call_args.args[1], root / '.experiment-state' / prefix)

    def test_both_build_search_generate_and_isolate_trace_state(self):
        from pipelines.server import server_client
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for prefix in ('a', 'b'):
                module = importlib.import_module('pipelines.server.server_run' if prefix == 'a' else 'pipelines.api.api_run')
                obs = importlib.import_module('observability.observability_trace')
                args = SimpleNamespace(command='build', processed_dir=root/prefix/'processed',
                    index_dir=root/prefix/'index', results_dir=root/prefix/'results', owner='test',
                    chunk_size=1000, chunk_overlap=150, top_k=1, question='예산?',
                    scenario_settings={'scenario':prefix.upper(), 'A_GENERATION_MODEL':'gen', 'A_EMBEDDING_MODEL':'emb'})
                module.write_json(args.processed_dir/'documents.json', [{'doc_id':'one', 'text':'예산은 100원입니다.', 'metadata':{'filename':'one.txt'}}])
                trace = obs.Trace({})
                with patch.dict(os.environ, {'OPENAI_API_KEY':'fake', 'OPENAI_GENERATION_MODEL':'gpt-5-mini',
                        'RETRIEVAL_REWRITE':'off', 'RETRIEVAL_RERANK':'none', 'RFP_CHUNKING_STRATEGY':'fixed', 'RFP_EMBEDDING_CONTEXT':'none'}), \
                        patch.object(module, 'OpenAI', return_value=FakeClient()), \
                        patch.object(server_client, 'ScenarioAClient', LocalFake), \
                        redirect_stdout(io.StringIO()), trace.run('split-test'):
                    module.run_pipeline(args, __import__('argparse').ArgumentParser(), {}, {'experiment_id':'test-0001'}, trace, 'test')
                    args.command='ask'
                    module.run_pipeline(args, __import__('argparse').ArgumentParser(), {}, {'experiment_id':'test-0001'}, trace, 'test')
                result=json.loads((args.results_dir/'ask_test-0001_test.json').read_text())
                self.assertEqual(result['sources'][0]['doc_id'], 'one')
            self.assertTrue((root/'a/index/index.faiss').is_file())
            self.assertTrue((root/'b/index/index.faiss').is_file())

    def test_both_pipelines_use_shared_implementations(self):
        server = importlib.import_module('pipelines.server.server_run')
        api = importlib.import_module('pipelines.api.api_run')
        for name in ('parse_documents', 'evaluate', 'Trace', 'next_experiment_id', 'save_report'):
            self.assertIs(getattr(server, name), getattr(api, name))
        self.assertFalse(list(ROOT.glob('a_*.py')))
        self.assertFalse(list(ROOT.glob('b_*.py')))
        self.assertFalse(list(ROOT.glob('common_*.py')))

    def test_run_selects_scenario_and_runs_its_own_interpreter(self):
        import run
        for scenario, value, module in (('A', 'A', 'pipelines.server.server_run'),
                                         ('B', '2', 'pipelines.api.api_run')):
            with patch.dict(os.environ, {'RFP_SCENARIO': value, 'RFP_PYTHON_A': 'py-a', 'RFP_PYTHON_B': 'py-b'}), \
                    patch.object(run.sys.stdin, 'isatty', return_value=False), \
                    patch('sys.argv', ['run.py', 'check-config']), patch.object(run.subprocess, 'call', return_value=0) as call:
                with self.assertRaises(SystemExit) as stop:
                    run.main()
            self.assertEqual(stop.exception.code, 0)
            self.assertEqual(call.call_args.args[0], ['py-' + scenario.lower(), '-m', module, 'check-config'])
            self.assertEqual(call.call_args.kwargs['env']['RFP_SCENARIO'], scenario)
            self.assertEqual(call.call_args.kwargs['cwd'], ROOT)

    def test_run_always_asks_in_terminal_and_uses_setting_as_default(self):
        import run
        for env, answer, expected in (({}, '3', 'AB'), ({'RFP_SCENARIO': 'B'}, '1', 'A'),
                                      ({'RFP_SCENARIO': 'A'}, '', 'A'), ({}, '', 'B')):
            with patch.dict(os.environ, env, clear=True), patch.object(run.sys.stdin, 'isatty', return_value=True), \
                    patch('builtins.input', return_value=answer) as asked, redirect_stdout(io.StringIO()):
                self.assertEqual(run.choose_scenario(), expected)
            asked.assert_called_once()
        with patch.dict(os.environ, {}, clear=True), patch.object(run.sys.stdin, 'isatty', return_value=False):
            self.assertEqual(run.choose_scenario(), 'B')

    def test_package_commands_work_in_fresh_processes(self):
        for module in ('pipelines.server.server_run', 'pipelines.api.api_run'):
            result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30, check=True)
            self.assertIn('check-config', result.stdout)

    def test_launcher_resolves_packages_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for scenario in ('A', 'B'):
                environment = {**os.environ, 'RFP_SCENARIO': scenario,
                               'RFP_PYTHON_A': sys.executable, 'RFP_PYTHON_B': sys.executable,
                               'A_GENERATION_MODEL': 'test-generation',
                               'A_EMBEDDING_MODEL': 'test-embedding'}
                result = subprocess.run([sys.executable, str(ROOT / 'run.py'), 'check-config'],
                                        cwd=directory, env=environment, capture_output=True,
                                        text=True, timeout=30, check=True)
                self.assertEqual(json.loads(result.stdout)['scenario'], scenario)

    def test_run_both_prints_one_summary_table(self):
        import run
        with tempfile.TemporaryDirectory() as directory:
            saved = {}
            for scenario, accuracy in (('A', 0.849), ('B', 0.996)):
                saved[scenario] = Path(directory) / f'{scenario}.json'
                saved[scenario].write_text(json.dumps({'summary': {'answer_accuracy': accuracy, 'recall_at_k': 1.0}}))
            out = io.StringIO()
            with patch.dict(os.environ, {'RFP_SCENARIO': 'AB'}), patch('sys.argv', ['run.py', 'evaluate']), \
                    patch.object(run.sys.stdin, 'isatty', return_value=False), \
                    patch.object(run, 'run_and_capture', side_effect=lambda s, args: str(saved[s])) as runner, \
                    redirect_stdout(out):
                run.main()
        self.assertEqual([c.args[0] for c in runner.call_args_list], ['A', 'B'])
        self.assertIn('| 답변 정확도 | 0.849 | 0.996 |', out.getvalue())

    def test_retrieval_scenario_does_not_follow_other_environment(self):
        from pipelines.server import server_retrieval
        from pipelines.api import api_retrieval
        with patch.dict(os.environ, {'RFP_SCENARIO': 'B', 'A_RERANK_MODEL': ''}, clear=True), \
                patch.object(server_retrieval, '_cross_encoder_available', return_value=True):
            self.assertEqual(server_retrieval._min_score(None), server_retrieval.DEFAULT_MIN_SCORE_A)
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(server_retrieval.retrieval_options()['rerank'], 'lexical')
        with patch.dict(os.environ, {'RFP_SCENARIO': 'A'}, clear=True):
            self.assertEqual(api_retrieval._min_score(None), api_retrieval.DEFAULT_MIN_SCORE)
            self.assertEqual(api_retrieval.rerank_model_source(), api_retrieval.DEFAULT_RERANK_MODEL)

    def test_evaluation_uses_its_own_trace(self):
        for prefix in ('a', 'b'):
            evaluation=importlib.import_module('evaluation.evaluation_metrics')
            obs=importlib.import_module('observability.observability_trace')
            trace=obs.Trace({})
            with trace.run('evaluation-isolation'):
                self.assertEqual(evaluation._usage_snapshot(), {})
                result=evaluation.evaluate([{'question':'예산?', 'expected_doc_ids':['one'], 'expected_keywords':['100원']}],
                    lambda question, filters: {'answer':'100원', 'sources':[], 'status':'ok'})
            self.assertEqual(result['summary']['keyword_coverage'], 1.0)
