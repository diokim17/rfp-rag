"""질문 재작성의 캐시·폴백·한도와 retrieve 연결을 외부 API 없이 검증합니다."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embedding import build_index, load_index
import query_rewrite
from query_rewrite import MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS, RewriteCache, rewrite_question
from retrieval import retrieve
from test_retrieval_rerank import FakeClient, document
from retrieval_baseline import setUpModule  # noqa: F401  검색 기본값을 baseline으로 고정


class FakeRewriteClient(FakeClient):
    """임베딩은 FakeClient, 재작성은 정해진 문장을 돌려줍니다. 호출 인자와 임베딩 입력을 기록합니다."""

    def __init__(self, output="학사정보시스템 고도화 요구사항", error=None):
        super().__init__()
        self.output, self.error, self.requests, self.embedded = output, error, [], []
        self.responses = SimpleNamespace(create=self.respond)

    def respond(self, **kwargs):
        self.requests.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.output, usage=SimpleNamespace(input_tokens=80, output_tokens=40))

    def embed(self, model, input):
        self.embedded += list(input)
        return super().embed(model, input)


class RewriteTests(unittest.TestCase):
    def setUp(self):
        RewriteCache._opened.clear()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name) / "rewrite.json"

    def test_caches_result_and_respects_output_limit(self):
        client = FakeRewriteClient()
        text, info = rewrite_question("가 사업 예산은?", client, "gpt-5-mini", self.cache)
        self.assertEqual((text, info["status"]), ("학사정보시스템 고도화 요구사항", "api"))
        self.assertEqual(client.requests[0]["max_output_tokens"], MAX_OUTPUT_TOKENS)
        self.assertEqual(client.requests[0]["model"], "gpt-5-mini")
        self.assertIs(client.requests[0]["store"], False)
        RewriteCache._opened.clear()  # 새 실행에서도 파일 캐시로 재사용
        again, info = rewrite_question("가 사업 예산은?", client, "gpt-5-mini", self.cache)
        self.assertEqual((again, info["status"]), (text, "cache"))
        self.assertEqual(len(client.requests), 1)
        rewrite_question("가 사업 예산은?", client, "gpt-5-nano", self.cache)  # 모델이 다르면 별도 항목
        self.assertEqual(len(client.requests), 2)

    def test_cache_file_has_no_api_key(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-secret-value"}):
            rewrite_question("가 사업 예산은?", FakeRewriteClient(), "gpt-5-mini", self.cache)
        content = self.cache.read_text(encoding="utf-8")
        self.assertNotIn("sk-test-secret-value", content)
        self.assertEqual(set(next(iter(json.loads(content).values()))), {"rewrite", "usage"})

    def test_falls_back_to_original_and_does_not_repeat_failures(self):
        for client, status in ((FakeRewriteClient(error=RuntimeError("down")), "fallback-error"),
                               (FakeRewriteClient(output="  "), "fallback-empty")):
            RewriteCache._opened.clear()
            text, info = rewrite_question("가 사업 예산은?", client, "gpt-5-mini", self.cache)
            self.assertEqual((text, info["status"]), ("가 사업 예산은?", status))
            text, info = rewrite_question("가 사업 예산은?", client, "gpt-5-mini", self.cache)
            self.assertEqual((text, info["status"]), ("가 사업 예산은?", "fallback-repeat"))
            self.assertEqual(len(client.requests), 1)  # 같은 실행 안에서 실패한 질문은 다시 호출하지 않음
        self.assertFalse(self.cache.exists())  # 실패는 캐시 파일에 남기지 않음

    def test_input_limit_blocks_call(self):
        client = FakeRewriteClient()
        long_question = "가" * (MAX_INPUT_TOKENS + 1)
        self.assertGreater(query_rewrite.estimate_input_tokens(long_question), MAX_INPUT_TOKENS)
        text, info = rewrite_question(long_question, client, "gpt-5-mini", self.cache)
        self.assertEqual((text, info["status"]), (long_question, "fallback-over-limit"))
        self.assertEqual(client.requests, [])


class RetrieveRewriteTests(unittest.TestCase):
    QUESTION = "나 사업의 요구사항"

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        build_index([
            document("bus", "버스정보시스템 구축", "가", "가 요구사항 목록과 제안 안내"),
            document("water", "용수공급 타당성조사", "나", "나 요구사항 목록과 제안 안내"),
            document("haksa", "학사정보시스템 고도화", "나", "다 요구사항 목록과 제안 안내"),
        ], FakeClient(), cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        RewriteCache._opened.clear()
        patcher = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("RETRIEVAL_")},
                             clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def search(self, client, **options):
        return retrieve(self.QUESTION, client, self.index, self.chunks, self.config, 3, **options)

    def test_default_never_calls_rewrite(self):
        client = FakeRewriteClient()
        self.search(client)
        self.assertEqual(client.requests, [])
        self.assertEqual(client.embedded, [self.QUESTION])

    def test_only_searches_rewritten_question(self):
        client = FakeRewriteClient(output="학사정보시스템 요구사항")
        hits = self.search(client, rewrite="only")
        self.assertEqual(client.embedded, ["학사정보시스템 요구사항"])
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(len(hits), 3)

    def test_both_searches_original_and_rewritten(self):
        client = FakeRewriteClient(output="학사정보시스템 요구사항")
        hits = self.search(client, rewrite="both", hybrid=True)
        self.assertEqual(client.embedded, [self.QUESTION, "학사정보시스템 요구사항"])
        self.assertTrue(all("rrf_score" in hit for hit in hits))

    def test_failure_falls_back_to_original_question(self):
        client = FakeRewriteClient(error=TimeoutError())
        hits = self.search(client, rewrite="only")
        self.assertEqual(client.embedded, [self.QUESTION])
        self.assertEqual([hit["doc_id"] for hit in hits], [hit["doc_id"] for hit in self.search(FakeClient())])
        with patch.dict(os.environ, {"RETRIEVAL_REWRITE": "only"}):
            self.search(client)  # 환경 변수로도 켜지고, 같은 실행의 실패는 다시 호출하지 않음
        self.assertEqual(len(client.requests), 1)


if __name__ == "__main__":
    unittest.main()
