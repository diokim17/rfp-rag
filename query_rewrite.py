"""김연주: 검색용 질문 재작성 (선택 기능, retrieve(rewrite=...)로 켤 때만 사용).

질문에 사업 유형·발주 기관·핵심 검색어를 보강해 한 줄로 다시 씁니다.
- 결과는 파일 캐시(질문·모델·프롬프트 버전의 해시 → 재작성문)에 저장해 같은 질문은 다시 호출하지 않습니다.
- 호출 실패, 빈 응답, 입력 한도 초과면 원문을 그대로 돌려줍니다(폴백). 실패는 캐시에 저장하지 않습니다.
- API 키는 OpenAI 클라이언트가 환경 변수(OPENAI_API_KEY)에서 읽으며 이 파일과 캐시에는 남기지 않습니다.
- 한도: 질문당 입력 약 250토큰(추정치로 사전 차단), 출력은 추론 포함 최대 500토큰(max_output_tokens).
"""

import hashlib
import json
import math
import re
from pathlib import Path

from observability import model_call

PROMPT_VERSION = 1
MAX_INPUT_TOKENS = 250
MAX_OUTPUT_TOKENS = 500
# 영어 지시문이 한국어보다 토큰이 적어 입력 한도를 지키기 쉽습니다. 출력은 한국어 한 줄입니다.
INSTRUCTIONS = ("Rewrite the Korean public-procurement RFP question below as one Korean search query. "
                "Keep its meaning. Add the likely project type, ordering agency and key terms only if implied. "
                "Never invent names, numbers or dates. Output only the query.")


def estimate_input_tokens(question):
    """보수적 추정: 한글·한자 1글자 = 1토큰, 그 밖의 글자는 3글자 = 1토큰, 지시문과 형식 여유 포함."""
    text = INSTRUCTIONS + "\n" + question
    wide = len(re.findall(r"[ㄱ-ㆎ가-힣一-鿿]", text))
    return wide + math.ceil((len(text) - wide) / 3) + 10


def cache_key(question, model):
    return hashlib.sha256(f"{PROMPT_VERSION}\n{model}\n{question}".encode()).hexdigest()


class RewriteCache:
    """질문 해시 → {"rewrite", "usage"}. 경로가 없으면 실행 중 메모리에만 둡니다 (같은 실행 안에서는 재사용)."""

    _opened = {}

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.items = json.loads(self.path.read_text(encoding="utf-8")) if self.path and self.path.is_file() else {}
        self.failed = set()  # 이번 실행에서 실패한 질문: 파일에는 남기지 않고 같은 실행 안에서 다시 호출하지 않음

    @classmethod
    def open(cls, path):
        key = str(Path(path).resolve()) if path else None  # 경로가 없으면 실행 중 메모리 캐시 하나를 공유
        if key not in cls._opened:
            cls._opened[key] = cls(path)
        return cls._opened[key]

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")


def rewrite_question(question, client, model="gpt-5-mini", cache_path=None):
    """반환: (검색에 쓸 질문, 정보). 정보의 status는 cache·api·fallback-error·fallback-empty·fallback-over-limit·fallback-repeat."""
    cache = RewriteCache.open(cache_path)
    key = cache_key(question, model)
    if key in cache.items:
        return cache.items[key]["rewrite"], {"status": "cache"}
    if key in cache.failed:
        return question, {"status": "fallback-repeat"}
    estimated = estimate_input_tokens(question)
    if estimated > MAX_INPUT_TOKENS:
        return question, {"status": "fallback-over-limit", "estimated_input_tokens": estimated}
    try:
        response = model_call("rewrite", model, lambda: client.responses.create(
            model=model, instructions=INSTRUCTIONS, input=question,
            reasoning={"effort": "minimal"}, max_output_tokens=MAX_OUTPUT_TOKENS, store=False,
        ), reasoning_effort="minimal", max_output_tokens=MAX_OUTPUT_TOKENS)
    except Exception as exc:  # noqa: BLE001 - 어떤 실패든 원문으로 계속 검색합니다.
        cache.failed.add(key)
        return question, {"status": "fallback-error", "error": type(exc).__name__,
                          "estimated_input_tokens": estimated}
    usage = getattr(response, "usage", None)
    usage = {name: getattr(usage, name, None) for name in ("input_tokens", "output_tokens")} if usage else {}
    text = " ".join((getattr(response, "output_text", "") or "").split())
    if not text:
        cache.failed.add(key)
        return question, {"status": "fallback-empty", "usage": usage, "estimated_input_tokens": estimated}
    cache.items[key] = {"rewrite": text, "usage": usage}
    cache.save()
    return text, {"status": "api", "usage": usage, "estimated_input_tokens": estimated}
