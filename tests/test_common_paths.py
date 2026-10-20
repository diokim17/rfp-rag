import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import common_paths


class ProjectPathTests(unittest.TestCase):
    def test_relative_paths_are_independent_of_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                self.assertEqual(common_paths.project_path("data/raw"), common_paths.ROOT / "data/raw")
            finally:
                os.chdir(previous)

    def test_model_defaults_and_relative_override_apply_to_all_kinds(self):
        for kind in ("generate", "embedding", "rerank"):
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(common_paths.model_directory(kind), common_paths.ROOT / "models" / kind)
            with patch.dict(os.environ, {"A_MODELS_DIR": "../shared-models"}, clear=True):
                self.assertEqual(common_paths.model_directory(kind), common_paths.ROOT / "../shared-models" / kind)

    def test_absolute_model_root_is_rejected(self):
        with patch.dict(os.environ, {"A_MODELS_DIR": "/tmp/models"}, clear=True):
            with self.assertRaisesRegex(ValueError, "상대 경로"):
                common_paths.model_directory("generate")
