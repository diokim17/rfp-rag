"""현재 고정 시나리오의 설정 검증. 이전 선택 로직은 legacy/tests에 보관합니다."""
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


class ScenarioSettingsTests(unittest.TestCase):
    def read(self, prefix, content, environment=None):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/('.env.'+('server' if prefix == 'a' else 'api'))
            path.write_text(content)
            with patch.dict(os.environ, environment or {}, clear=True):
                return importlib.import_module('pipelines.server.server_run' if prefix == 'a' else 'pipelines.api.api_run').read_scenario_settings(path)

    def test_fixed_scenario_ignores_other_selector(self):
        for prefix in ('a','b'):
            settings=self.read(prefix, 'SERVER_GENERATION_MODEL=gen\nSERVER_EMBEDDING_MODEL=emb\n', {'RFP_SCENARIO':'B' if prefix=='a' else 'A'})
            self.assertEqual(settings['scenario'], prefix.upper())

    def test_a_requires_both_model_names_and_rejects_paths(self):
        for key in ('SERVER_GENERATION_MODEL','SERVER_EMBEDDING_MODEL'):
            for value in ('', '.', '..', '../model', '/tmp/model', 'org/model', 'org\\model'):
                with self.subTest(key=key,value=value), self.assertRaisesRegex(ValueError,key):
                    self.read('a', f'SERVER_GENERATION_MODEL=gen\nSERVER_EMBEDDING_MODEL=emb\n{key}={value}\n')

    def test_existing_environment_takes_precedence(self):
        self.assertEqual(self.read('a','SERVER_GENERATION_MODEL=gen\nSERVER_EMBEDDING_MODEL=emb\n', {'SERVER_GENERATION_MODEL':'other'})['SERVER_GENERATION_MODEL'],'other')

    def test_check_config_stops_before_models_and_hides_secrets(self):
        for prefix in ('a','b'):
            module=importlib.import_module('pipelines.server.server_run' if prefix == 'a' else 'pipelines.api.api_run')
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                (root/('.env.'+('server' if prefix == 'a' else 'api'))).write_text('SERVER_GENERATION_MODEL=gen\nSERVER_EMBEDDING_MODEL=emb\nOPENAI_API_KEY=SECRET_TEST_VALUE\n')
                output=io.StringIO()
                with patch.dict(os.environ,{},clear=True), patch.object(module,'ROOT',root), \
                        patch('sys.argv',['server_run.py' if prefix == 'a' else 'api_run.py','check-config']), \
                        patch.object(module,'run_pipeline') as pipeline, \
                        patch.object(module,'next_experiment_id') as experiment, redirect_stdout(output):
                    module.main()
                self.assertEqual(json.loads(output.getvalue())['scenario'],prefix.upper())
                self.assertNotIn('SECRET_TEST_VALUE',output.getvalue())
                pipeline.assert_not_called()
                experiment.assert_not_called()
