"""A 가짜 담당 모듈로 실제 FAISS 구축·검색·생성 연결을 검증합니다."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import run
from observability import Trace
from scenario_a import ScenarioAClient


class ScenarioConnectionTests(unittest.TestCase):
    def test_a_build_ask_and_evaluate_reuse_models_without_openai(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gen, emb = root / 'generate', root / 'embedding'
            (gen / 'gen').mkdir(parents=True)
            (emb / 'emb').mkdir(parents=True)
            embedding = SimpleNamespace(embed_texts=Mock(side_effect=lambda texts: [[3., 4.] for _ in texts]))
            def answer(question, hits, model):
                return {'question': question, 'answer': '예산 답변', 'model': model,
                        'sources': [{'citation': i, **hit} for i, hit in enumerate(hits, 1)]}
            generation = SimpleNamespace(generate_answer=Mock(side_effect=answer))
            emb_factory = Mock(return_value=embedding)
            gen_factory = Mock(return_value=generation)
            modules = {'scenario_a_embedding': SimpleNamespace(load_model=emb_factory),
                       'scenario_a_generation': SimpleNamespace(load_model=gen_factory)}
            settings = {'scenario': 'A', 'A_GENERATION_MODEL': 'gen', 'A_EMBEDDING_MODEL': 'emb'}
            args = SimpleNamespace(command='all', scenario_settings=settings, raw_dir=root/'raw',
                processed_dir=root/'processed', index_dir=root/'index', results_dir=root/'results',
                reports_dir=root/'reports', limit=1, owner='test', chunk_size=1000,
                chunk_overlap=150, top_k=1, question='예산?', eval_file=root/'eval.json')
            docs = [{'doc_id': 'a', 'text': '예산 안내', 'metadata': {'filename': 'a.txt'}}]
            def parse(raw, processed, limit):
                run.write_json(processed/'documents.json', docs)
                run.write_json(processed/'parsing_errors.json', [])
                return docs
            with patch.dict(os.environ, {'RETRIEVAL_REWRITE': 'off', 'RETRIEVAL_RERANK': 'none',
                                         'RETRIEVAL_HYBRID': 'false'}, clear=True), \
                    patch.dict('sys.modules', modules), \
                    patch.object(run, 'A_GENERATION_MODEL_DIR', gen), \
                    patch.object(run, 'A_EMBEDDING_MODEL_DIR', emb), \
                    patch('run.parse_documents', side_effect=parse), \
                    patch('run.OpenAI') as openai, redirect_stdout(io.StringIO()):
                run.run_pipeline(args, run.argparse.ArgumentParser(), {}, {'experiment_id':'test-0001'}, Trace({}), 'test')
                emb_factory.assert_called_once_with(emb/'emb')
                gen_factory.assert_called_once_with(gen/'gen')
                self.assertGreaterEqual(embedding.embed_texts.call_count, 2)
                generation.generate_answer.assert_called_once()
                config = run.read_json(args.index_dir/'config.json')
                self.assertEqual(config['embedding_model'], 'emb')
                result = run.read_json(args.results_dir/'all_test-0001_test.json')
                self.assertEqual(result['model'], 'gen')
                self.assertEqual(result['sources'][0]['doc_id'], 'a')
                openai.assert_not_called()
                args.command = 'evaluate'
                run.write_json(args.eval_file, [{'question':'예산?', 'expected_doc_ids':['a']},
                                               {'question':'예산 안내?', 'expected_doc_ids':['a']}])
                emb_factory.reset_mock(); gen_factory.reset_mock()
                run.run_pipeline(args, run.argparse.ArgumentParser(), {}, {'experiment_id':'test-0002'}, Trace({}), 'test')
                emb_factory.assert_called_once(); gen_factory.assert_called_once()
                self.assertEqual(generation.generate_answer.call_count, 3)

    def test_main_uses_a_defaults_and_respects_explicit_paths(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'.env').write_text('RFP_SCENARIO=A\nA_GENERATION_MODEL=gen\nA_EMBEDDING_MODEL=emb\n')
                argv = ['run.py', 'build']
                if explicit:
                    argv += ['--index-dir', str(root/'custom'), '--results-dir='+str(root/'output')]
                with patch.dict(os.environ, {}, clear=True), patch.object(run, 'ROOT', root), \
                        patch('sys.argv', argv), patch('run.owner_initials', return_value='test'), \
                        patch('run.next_experiment_id', return_value='test-0001'), \
                        patch('run.run_pipeline') as pipeline, redirect_stdout(io.StringIO()):
                    run.main()
                args = pipeline.call_args.args[0]
                self.assertEqual(args.index_dir, root/('custom' if explicit else 'indexes/scenario-a'))
                self.assertEqual(args.results_dir, root/('output' if explicit else 'results/scenario-a'))
                self.assertEqual(args.scenario_settings['scenario'], 'A')
