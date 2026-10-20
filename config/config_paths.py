import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value, root=None):
    path = Path(value)
    return path if path.is_absolute() else (ROOT if root is None else root) / path


def model_directory(kind, default=None):
    configured = os.getenv("A_MODELS_DIR", "").strip()
    if configured:
        path = Path(configured)
        if path.is_absolute():
            raise ValueError("A_MODELS_DIR에는 프로젝트 기준 상대 경로를 입력하세요. 예: models, ../shared-models")
        return project_path(path) / kind
    return Path(default) if default is not None else ROOT / "models" / kind
