"""시나리오 A: 서버에 저장된 모델로 문서·질문을 숫자 벡터로 변환한다.

호출 흐름:
    ScenarioAClient → load_model(모델 폴더) → 반환 객체.embed_texts(문자열 목록)
    → a_embedding.py에서 벡터 검증·L2 정규화 → 문서 인덱스 구축 또는 질문 검색

모델 다운로드는 별도 준비 단계에서 수행한다. 이 파일을 import하는 것만으로는
.env를 읽거나 모델을 다운로드·로딩하지 않는다. 호출자가 load_model의 반환 객체를
보관하고 재사용하므로, embed_texts를 호출할 때마다 모델을 다시 읽지 않는다.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common_observability import model_call, observed


class _LocalEmbedding:
    """로딩된 모델과 실행 통계를 보관하는 내부 객체.

    외부에서는 embed_texts(texts)만 호출하면 된다. 클래스 이름 앞의 밑줄은
    이 클래스를 직접 생성하기보다 load_model을 통해 사용하라는 의미다.
    """

    def __init__(self, model, torch, model_name):
        self.model, self.torch = model, torch
        # 추적 기록에서 로컬 모델임을 구분한다. 실제 모델 폴더 이름은 바꾸지 않는다.
        self.model_name = "local/" + model_name
        # 한 번에 추론할 텍스트 수. GPU 메모리가 부족하면 줄인 값을 계속 사용한다.
        self.batch_size = 8
        # calls/texts/input_tokens/truncated_texts는 전체 요청이 성공했을 때 누적한다.
        # oom_retries는 배치 축소 횟수, failures는 최종적으로 실패한 추론 요청 수다.
        self.stats = dict(calls=0, texts=0, input_tokens=0, truncated_texts=0,
                          oom_retries=0, failures=0)

    def embed_texts(self, texts):
        """문자열 목록 → 입력 순서의 float32 배열, 형태는 (텍스트 수, 벡터 차원).

        예: 텍스트 3개를 BGE-M3로 처리하면 (3, 1024) 배열을 반환한다.
        빈 목록과 비문자열 원소는 거부하지만, 빈 문자열 자체는 허용한다.
        문서와 질문 모두 같은 모델·처리 방식을 사용한다.
        """
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("texts must be a non-empty list of strings")
        if not all(isinstance(text, str) for text in texts):
            raise ValueError("texts must contain only strings")
        try:
            # 기존 추적 도구가 시간·모델 설정·토큰 수를 기록한다.
            # texts는 실행 함수 안에서만 사용하고 추적 메타데이터로 넘기지 않는다.
            response = model_call("embedding", self.model_name,
                                  lambda: self._encode(texts), batch_size=self.batch_size,
                                  max_length=self.model.max_seq_length, local=True)
        except Exception:
            self.stats["failures"] += 1
            # 오류를 숨기거나 다른 모델로 대체하지 않고 호출자에게 그대로 전달한다.
            raise
        # 내부 추적용 usage 정보는 감추고, 공통 임베딩 코드에는 배열만 반환한다.
        return response.vectors

    def _encode(self, texts):
        """텍스트를 작은 배치로 추론하고, 벡터와 추적용 토큰 수를 함께 반환한다."""
        vectors, tokens, truncated, start = [], 0, 0, 0
        while start < len(texts):
            batch = texts[start:start + self.batch_size]
            # 토큰 수를 세기 위한 전처리다. 실제 SentenceTransformer의 공백 제거와
            # 선택적 소문자 변환에 맞춘다. 추론 자체는 아래 model.encode가 처리한다.
            prepared = [text.strip() for text in batch]
            if getattr(self.model[0], "do_lower_case", False):
                prepared = [text.lower() for text in prepared]
            # 길이 초과 여부를 알아야 하므로 여기서는 자르지 않는다.
            # 특수 토큰은 포함하고 padding은 제외한다. 본문·토큰 ID는 기록하지 않는다.
            lengths = [len(ids) for ids in self.model.tokenizer(
                prepared, padding=False, truncation=False, verbose=False)["input_ids"]]
            try:
                # 추론 시에는 학습용 기울기를 계산하지 않아 메모리 사용을 줄인다.
                with self.torch.inference_mode():
                    # 접두어를 추가하지 않고 모델에 저장된 pooling 구성을 사용한다.
                    # CPU NumPy 배열을 받는다. encode 옵션의 추가 정규화는 끄지만,
                    # 모델 내부 Normalize 계층은 실행될 수 있다. 최종 검증·정규화는
                    # 공통 a_embedding.py에서 수행한다.
                    encoded = self.model.encode(
                        batch, batch_size=len(batch), prompt="", show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=False)
            except self.torch.cuda.OutOfMemoryError:
                if len(batch) == 1:
                    # 한 개도 처리할 수 없으면 배치 축소로 해결할 수 없으므로 종료한다.
                    raise
                self.batch_size = max(1, len(batch) // 2)
                self.stats["oom_retries"] += 1
                # 사용하지 않는 GPU 캐시를 반환한다. 모델 가중치를 내리지는 않는다.
                self.torch.cuda.empty_cache()
                # start를 옮기지 않아 실패한 위치부터 더 작은 배치로 다시 처리한다.
                continue
            # 성공한 배치만 저장하고 입력 위치를 전진시켜 순서와 개수를 보존한다.
            vectors.append(np.asarray(encoded, dtype=np.float32))
            # 실제 추론은 최대 길이에서 잘리므로 처리 토큰 수도 그 상한까지만 센다.
            tokens += sum(min(n, self.model.max_seq_length) for n in lengths)
            truncated += sum(n > self.model.max_seq_length for n in lengths)
            start += len(batch)
        # 배치별 (배치 크기, 벡터 차원) 배열을 행 방향으로 이어 붙인다.
        result = np.concatenate(vectors, axis=0)
        self.stats["calls"] += 1
        self.stats["texts"] += len(texts)
        self.stats["input_tokens"] += tokens
        self.stats["truncated_texts"] += truncated
        # 기존 model_call이 response.usage에서 토큰 수를 읽을 수 있게 맞춘 내부 형식.
        # 임베딩은 문장을 생성하지 않으므로 출력 토큰은 0이다. API 과금량은 아니다.
        return SimpleNamespace(vectors=result, usage=SimpleNamespace(
            input_tokens=tokens, output_tokens=0, total_tokens=tokens))


@observed("load-local-embedding")
def load_model(model_path: Path):
    """준비된 로컬 모델을 읽고 embed_texts(texts)를 가진 객체를 반환한다.

    model_path 예: Path('models/embedding/default_model')
    .env의 모델 폴더 이름을 실제 경로로 합치는 작업은 ScenarioAClient가 담당한다.
    위 observed 데코레이터는 입력 본문 없이 모델 로딩 단계의 시간·오류를 추적한다.
    """
    path = Path(model_path)
    # modules.json은 Transformer·Pooling·Normalize 등 모델 구성 순서를 정의한다.
    if not path.is_dir() or not (path / "modules.json").is_file():
        raise OSError("A local SentenceTransformer model with modules.json is required")
    # 선택 의존성은 실제 A 모델을 로딩할 때만 import한다.
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # GPU에서는 FP16으로 메모리를 절약하고, CPU에서는 FP32를 사용한다.
    # 로컬 파일만 읽으며, 모델 저장소의 사용자 정의 코드는 실행하지 않는다.
    model = SentenceTransformer(
        str(path.resolve()), device=device, local_files_only=True, trust_remote_code=False,
        model_kwargs={"dtype": torch.float16 if device == "cuda" else torch.float32})
    # 모델 자체의 길이 제한을 넘지 않으면서 최대 8192토큰까지 허용한다.
    model.max_seq_length = min(8192, model.max_seq_length)
    # dropout 등 학습 전용 동작을 끄고 평가·추론 모드로 전환한다.
    model.eval()
    return _LocalEmbedding(model, torch, path.name)
