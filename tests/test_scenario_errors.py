"""A 준비 실패 시 원인을 안내하고 B/API 실행 없이 종료하는지 검증합니다."""
import argparse
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pipelines.server.server_run as run
from observability.observability_trace import Trace


class ScenarioErrorsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.gen = self.root / 'generate'
        self.emb = self.root / 'embedding'
        (self.gen/'gen').mkdir(parents=True)
        (self.emb/'emb').mkdir(parents=True)
        self.args = SimpleNamespace(command='ask', scenario_settings={
            'scenario':'A', 'SERVER_GENERATION_MODEL':'gen', 'SERVER_EMBEDDING_MODEL':'emb'})
        self.modules = {
            'pipelines.server.server_local_embedding': SimpleNamespace(load_model=Mock(return_value=SimpleNamespace(embed_texts=Mock()))),
            'pipelines.server.server_local_generation': SimpleNamespace(load_model=Mock(return_value=SimpleNamespace(generate_answer=Mock()))),
        }

    def reject(self, message, rewrite='off'):
        output = io.StringIO()
        with patch.dict(os.environ, {'OPENAI_API_KEY':'fake', 'RETRIEVAL_REWRITE':rewrite}, clear=True), \
                patch.object(run, 'SERVER_GENERATION_MODEL_DIR', self.gen), \
                patch.object(run, 'SERVER_EMBEDDING_MODEL_DIR', self.emb), \
                patch.dict('sys.modules', self.modules), \
                patch('pipelines.server.server_run.check_index_documents'), patch('pipelines.server.server_run.OpenAI') as client, \
                patch('pipelines.server.server_run.build_index') as build, patch('pipelines.server.server_run.retrieve') as retrieve, \
                patch('pipelines.server.server_run.generate_answer') as generate, \
                redirect_stderr(output), self.assertRaises(SystemExit) as error:
            run.run_pipeline(self.args, argparse.ArgumentParser(), {}, {}, Trace({}), 'test')
        self.assertEqual(error.exception.code, 2)
        self.assertIn(message, output.getvalue())
        for call in (client, build, retrieve, generate):
            call.assert_not_called()

    def test_missing_embedding_and_generation_folders(self):
        for field in ('SERVER_EMBEDDING_MODEL', 'SERVER_GENERATION_MODEL'):
            with self.subTest(field=field):
                self.args.scenario_settings[field] = 'missing'
                self.reject('모델 폴더가 없습니다')
                self.args.scenario_settings[field] = 'emb' if field == 'SERVER_EMBEDDING_MODEL' else 'gen'

    def test_missing_each_module(self):
        for name in self.modules:
            with self.subTest(name=name):
                with patch('pipelines.server.server_client.import_module', side_effect=lambda n: self.modules[n] if n != name else self.missing(n)):
                    self.reject(name)

    @staticmethod
    def missing(name):
        raise ModuleNotFoundError(f'No module named {name}', name=name)

    def test_missing_dependency_identified(self):
        with patch('pipelines.server.server_client.import_module', side_effect=ModuleNotFoundError('missing torch', name='torch')):
            self.reject('torch')

    def test_missing_factory_and_method(self):
        for name, method in (('pipelines.server.server_local_embedding', 'embed_texts'), ('pipelines.server.server_local_generation', 'generate_answer')):
            original = self.modules[name]
            for module, message in ((SimpleNamespace(), 'load_model'),
                                    (SimpleNamespace(load_model=Mock(return_value=object())), method)):
                with self.subTest(name=name, message=message):
                    self.modules[name] = module
                    self.reject(message)
            self.modules[name] = original

    def test_unimplemented_loader_and_missing_weights(self):
        for name in self.modules:
            original = self.modules[name]
            for error, message in ((NotImplementedError(), '미구현'),
                                   (FileNotFoundError('weights.safetensors'), '모델 파일을 로딩할 수 없습니다')):
                with self.subTest(name=name, error=type(error).__name__):
                    self.modules[name] = SimpleNamespace(load_model=Mock(side_effect=error))
                    self.reject(message)
            self.modules[name] = original

    def test_a_rejects_openai_query_rewrite(self):
        self.reject('RETRIEVAL_REWRITE=off', rewrite='both')
