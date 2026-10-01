"""선택적 Langfuse 추적. 원문/질문/답변/키는 외부로 보내지 않습니다."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import os
import subprocess
import warnings
from urllib.parse import urlsplit

_active = ContextVar("rfp_trace", default=None)


def code_version(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL, timeout=5).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


class Trace:
    def __init__(self, metadata, client=None):
        self.metadata = metadata
        self.client = client
        self.trace_id = None
        self.usage = {}

    @classmethod
    def from_env(cls, metadata):
        if os.getenv("LANGFUSE_ENABLED", "true").lower() in {"false", "0", "no"}:
            return cls(metadata)
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret = os.getenv("LANGFUSE_SECRET_KEY")
        if not public and not secret:
            return cls(metadata)
        url = os.getenv("LANGFUSE_BASE_URL", "").strip()
        parsed = urlsplit(url)
        if not public or not secret or parsed.scheme not in {"http", "https"} or not parsed.netloc or any(c in url for c in "[]()"):
            warnings.warn("Langfuse 설정이 불완전합니다. BASE_URL은 Markdown 링크가 아닌 URL 문자열이어야 합니다. 로컬 기록만 저장합니다.")
            return cls(metadata)
        try:
            from langfuse import Langfuse
            return cls(metadata, Langfuse(public_key=public, secret_key=secret,
                                          base_url=url, timeout=5))
        except Exception:
            warnings.warn("Langfuse 초기화 실패: 로컬 기록만 저장합니다.")
            return cls(metadata)

    @contextmanager
    def span(self, name, **kwargs):
        if self.client is None:
            yield None
            return
        # SDK 오류만 무시합니다. 파이프라인 오류는 원래대로 호출자에게 전달합니다.
        try:
            manager = self.client.start_as_current_observation(name=name, **kwargs)
            observation = manager.__enter__()
        except Exception:
            warnings.warn("Langfuse 단계 기록 시작 실패: 실행을 계속합니다.")
            yield None
            return
        try:
            yield observation
        except BaseException as exc:
            self.update(observation, level="ERROR", status_message=type(exc).__name__)
            raise
        finally:
            try:
                # SDK의 자동 예외 메시지 수집을 피하고 오류 종류만 기록합니다.
                manager.__exit__(None, None, None)
            except Exception:
                warnings.warn("Langfuse 단계 기록 종료 실패: 실행을 계속합니다.")

    def update(self, observation, **kwargs):
        if observation is not None:
            try:
                observation.update(**kwargs)
            except Exception:
                warnings.warn("Langfuse 기록 갱신 실패: 실행을 계속합니다.")

    @contextmanager
    def run(self, name):
        token = _active.set(self)
        try:
            with self.span(name, metadata=self.metadata) as observation:
                if observation is not None:
                    self.trace_id = observation.trace_id
                yield self
        finally:
            _active.reset(token)
            if self.client is not None:
                try:
                    self.client.flush()
                except Exception:
                    warnings.warn("Langfuse 전송 실패: 로컬 결과를 확인하세요.")

    def scores(self, values):
        if self.client is not None and self.trace_id:
            for name, value in values.items():
                if value is not None:
                    try:
                        self.client.create_score(trace_id=self.trace_id, name=name,
                                                 value=float(value), data_type="NUMERIC")
                    except Exception:
                        warnings.warn("Langfuse 평가 점수 기록 실패: 로컬 결과를 확인하세요.")


def observed(name):
    """입출력 본문을 자동 수집하지 않는 단계별 추적."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            trace = _active.get()
            if trace is None:
                return fn(*args, **kwargs)
            with trace.span(name):
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def record_retrieval(hits):
    trace = _active.get()
    if trace is not None:
        with trace.span("retrieval-results", output=[{
                "doc_id": h["doc_id"], "chunk_id": h["chunk_id"], "score": h["score"]
        } for h in hits]):
            pass


def model_call(kind, model, call, **parameters):
    """명시적으로 선별한 모델 설정/토큰만 기록; API 요청 본문은 수집하지 않음."""
    trace = _active.get()
    if trace is None:
        return call()
    with trace.span(kind, as_type=kind, model=model, model_parameters=parameters) as observation:
        response = call()
        usage = getattr(response, "usage", None)
        details = {}
        if usage is not None:
            for target, candidates in {"input": ("input_tokens", "prompt_tokens"),
                                       "output": ("output_tokens", "completion_tokens"),
                                       "total": ("total_tokens",)}.items():
                for field in candidates:
                    value = getattr(usage, field, None)
                    if value is not None:
                        details[target] = int(value)
                        break
        totals = trace.usage.setdefault(model, {})
        for key, value in details.items():
            totals[key] = totals.get(key, 0) + value
        trace.update(observation, usage_details=details)
        return response
