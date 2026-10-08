"""A 담당 모듈 연결. 실제 모델 다운로드·추론 구현은 담당 모듈에 둡니다."""
from importlib import import_module


def load_component(module_name, path, method):
    if not path.is_dir():
        raise ValueError(f"시나리오 A 모델 폴더가 없습니다: {path}")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(f"시나리오 A 모듈 또는 의존성이 없습니다: {exc.name}") from exc
    factory = getattr(module, "load_model", None)
    if not callable(factory):
        raise ValueError(f"{module_name}.load_model(model_path) 구현이 필요합니다.")
    try:
        component = factory(path)
    except NotImplementedError as exc:
        raise ValueError(f"시나리오 A 모델 로딩이 미구현입니다: {module_name}.load_model ({path})") from exc
    except OSError as exc:
        raise ValueError(f"시나리오 A 모델 파일을 로딩할 수 없습니다: {module_name} ({path}): {exc}") from exc
    if not callable(getattr(component, method, None)):
        raise ValueError(f"{module_name}.load_model 반환 객체에 {method} 구현이 필요합니다.")
    return component


class ScenarioAClient:
    """기존 client 인자를 유지하고, 모델 객체는 실행 중 재사용합니다.

    scenario_a_embedding.load_model(Path) -> embed_texts(texts) 메서드가 있는 객체
    scenario_a_generation.load_model(Path) -> generate_answer(question, hits, model) 객체
    임베딩은 입력 순서의 2차원 벡터, 생성은 기존 답변 dict를 반환합니다.
    """
    def __init__(self, settings, generation_dir, embedding_dir, *, need_generation=True):
        self.embedding_model = settings["A_EMBEDDING_MODEL"]
        self.generation_model = settings["A_GENERATION_MODEL"]
        self.embedding_path = embedding_dir / self.embedding_model
        self.generation_path = generation_dir / self.generation_model
        self.embedding = load_component("scenario_a_embedding", self.embedding_path, "embed_texts")
        self.generation = (load_component("scenario_a_generation", self.generation_path, "generate_answer")
                           if need_generation else None)

    def embed_texts(self, texts, model):
        if model != self.embedding_model:
            raise ValueError("선택한 A 임베딩 모델과 인덱스 모델이 다릅니다. A 인덱스를 다시 build하세요.")
        return self.embedding.embed_texts(texts)

    def generate_answer(self, question, hits, model):
        return self.generation.generate_answer(question, hits, model)
