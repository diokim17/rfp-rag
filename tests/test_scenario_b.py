"""B의 기본 경로와 OpenAI 호출 흐름을 실제 API 없이 검증합니다."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

import run
from observability import Trace
from test_pipeline import FakeClient


class ScenarioBTests(TestCase):
    def test_b_and_unset_scenario_keep_original_paths(self):
        for scenario in ('', 'RFP_SCENARIO=B\n'):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'.env').write_text(scenario)
                with patch.dict(os.environ, {}, clear=True), patch.object(run, 'ROOT', root), \
                        patch('sys.argv', ['run.py','build']), \
                        patch('run.owner_initials', return_value='test'), \
                        patch('run.next_experiment_id', return_value='test-0001'), \
                        patch('run.run_pipeline') as pipeline, redirect_stdout(io.StringIO()):
                    run.main()
                args = pipeline.call_args.args[0]
                self.assertEqual(args.scenario_settings['scenario'], 'B')
                self.assertEqual(args.index_dir, root/'indexes')
                self.assertEqual(args.results_dir, root/'results')
                self.assertEqual(args.reports_dir, root/'results/reports')

    def test_b_build_ask_evaluate_preserve_models_and_results(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(command='build', scenario_settings={'scenario':'B'},
                processed_dir=root/'processed', index_dir=root/'indexes', results_dir=root/'results',
                reports_dir=root/'reports', owner='test', chunk_size=1000, chunk_overlap=150,
                top_k=1, question='예산?', eval_file=root/'eval.json')
            run.write_json(args.processed_dir/'documents.json', [
                {'doc_id':'a', 'text':'예산 100원', 'metadata':{'filename':'a.txt'}}])
            run.write_json(args.eval_file, [{'question':'예산?', 'expected_doc_ids':['a'],
                                            'expected_keywords':['100원']}])
            fake = FakeClient()
            with patch.dict(os.environ, {'OPENAI_API_KEY':'fake', 'OPENAI_GENERATION_MODEL':'gpt-5-nano',
                    'OPENAI_EMBEDDING_MODEL':'test-embedding', 'RETRIEVAL_RERANK':'none',
                    'RETRIEVAL_HYBRID':'false', 'RETRIEVAL_REWRITE':'off'}, clear=True), \
                    patch('run.OpenAI', return_value=fake) as client, \
                    patch('scenario_a.load_component') as a_loader, redirect_stdout(io.StringIO()):
                for command in ('build','ask','evaluate'):
                    args.command = command
                    run.run_pipeline(args, run.argparse.ArgumentParser(), {},
                                     {'experiment_id':'test-0001'}, Trace({}), 'test')
                self.assertEqual(client.call_count, 3)
                a_loader.assert_not_called()
            self.assertEqual(set(fake.models), {'test-embedding'})
            answer = run.read_json(args.results_dir/'ask_test-0001_test.json')
            self.assertEqual(answer['model'], 'gpt-5-nano')
            self.assertEqual(answer['status'], 'ok')
            self.assertEqual(answer['sources'][0]['doc_id'], 'a')
            report = run.read_json(args.results_dir/'evaluate_test-0001_test.json')
            self.assertEqual(report['summary']['recall_at_k'], 1.)
            self.assertEqual(report['summary']['keyword_coverage'], 1.)
