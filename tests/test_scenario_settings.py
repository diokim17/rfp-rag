"""실제 모델·API 없이 .env 설정 읽기와 실행 진입점을 검증합니다."""
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import run


class ScenarioSettingsTests(unittest.TestCase):
    def read(self, content, environment=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / '.env'
            path.write_text(content, encoding='utf-8')
            with patch.dict(os.environ, environment or {}, clear=True):
                return run.read_scenario_settings(path)

    def test_default_b_and_empty_model_names(self):
        self.assertEqual(self.read('')['scenario'], 'B')
        self.assertEqual(self.read('RFP_SCENARIO=B\nA_GENERATION_MODEL=\n')['A_GENERATION_MODEL'], '')

    def test_a_reads_both_folder_names(self):
        settings = self.read('RFP_SCENARIO=A\nA_GENERATION_MODEL=model-a\nA_EMBEDDING_MODEL=model-b\n')
        self.assertEqual(settings, {'scenario': 'A', 'A_GENERATION_MODEL': 'model-a',
                                    'A_EMBEDDING_MODEL': 'model-b'})

    def test_normalizes_scenario_and_whitespace(self):
        self.assertEqual(self.read('RFP_SCENARIO=" a "\nA_GENERATION_MODEL=" model-a "\nA_EMBEDDING_MODEL=model-b')['scenario'], 'A')

    def test_invalid_scenario_is_rejected(self):
        for value in ('', 'C'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'RFP_SCENARIO'):
                self.read(f'RFP_SCENARIO={value}\n')

    def test_a_requires_each_model_name(self):
        for key in ('A_GENERATION_MODEL', 'A_EMBEDDING_MODEL'):
            content = 'RFP_SCENARIO=A\nA_GENERATION_MODEL=gen\nA_EMBEDDING_MODEL=emb\n'
            content += f'{key}=\n'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.read(content)

    def test_paths_cannot_escape_shared_model_directory(self):
        for key in ('A_GENERATION_MODEL', 'A_EMBEDDING_MODEL'):
            for value in ('.', '..', '../model', '/tmp/model', 'org/model', 'org\\model'):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    self.read(f'RFP_SCENARIO=A\nA_GENERATION_MODEL=gen\nA_EMBEDDING_MODEL=emb\n{key}={value}\n')

    def test_existing_environment_takes_precedence(self):
        self.assertEqual(self.read('RFP_SCENARIO=A', {'RFP_SCENARIO': 'B'})['scenario'], 'B')

    def test_check_config_stops_before_pipeline_and_prints_no_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.env').write_text('RFP_SCENARIO=A\nA_GENERATION_MODEL=gen\nA_EMBEDDING_MODEL=emb\nOPENAI_API_KEY=SECRET_TEST_VALUE\n')
            output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True), patch.object(run, 'ROOT', root), \
                    patch('sys.argv', ['run.py', 'check-config']), \
                    patch('run.run_pipeline') as pipeline, patch('run.next_experiment_id') as experiment, \
                    patch('run.OpenAI') as client, redirect_stdout(output):
                run.main()
            self.assertEqual(json.loads(output.getvalue())['scenario'], 'A')
            self.assertNotIn('SECRET_TEST_VALUE', output.getvalue())
            pipeline.assert_not_called()
            experiment.assert_not_called()
            client.assert_not_called()


if __name__ == '__main__':
    unittest.main()
