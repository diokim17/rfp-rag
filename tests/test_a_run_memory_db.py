"""박단비: a_run을 불러오면 GPU 메모리 설정(expandable_segments)이 기본으로 켜지는지 확인합니다."""

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = "import os; import a_run; print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))"


class GpuMemorySettingTests(unittest.TestCase):
    def run_check(self, value=None):
        env = {k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"}
        if value:
            env["PYTORCH_CUDA_ALLOC_CONF"] = value
        # 새 프로세스에서 확인해야 다른 테스트가 이미 불러온 a_run의 영향을 받지 않습니다.
        return subprocess.run([sys.executable, "-c", CHECK], cwd=ROOT, env=env, capture_output=True, text=True,
                              check=True).stdout.strip()

    def test_default_is_expandable_segments(self):
        self.assertEqual(self.run_check(), "expandable_segments:True")

    def test_existing_value_is_kept(self):
        self.assertEqual(self.run_check("max_split_size_mb:128"), "max_split_size_mb:128")


if __name__ == "__main__":
    unittest.main()
