"""박단비: 검색 추천 설정(하이브리드 + lexical + 문서당 상한 2)의 결과가 생성 함수에 몇 개, 어떤 순서로
들어오는지와, top_k보다 적게 들어와도 오류 없이 답변이 만들어지는지 외부 API 없이 검증합니다."""

import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import openai

from embedding import build_index, load_index
from generation import generate_answer
from retrieval import retrieve

class FakeClient:
    """본문 첫 글자로 질문과의 코사인 순서(가 > 나 > 다 > 그 외)를 정합니다.

    다른 담당자의 테스트 파일이 바뀌어도 이 테스트가 깨지지 않도록 이 파일 안에 따로 둡니다.
    """

    CLOSENESS = {"학": 1., "가": .9, "나": .8, "다": .7}  # '학'은 질문의 첫 글자

    def __init__(self):
        self.embeddings = SimpleNamespace(create=self.embed)
        self.responses = SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text="요구사항입니다. [1]"))

    def embed(self, model, input):
        closeness = [self.CLOSENESS.get(text[0], 0.) for text in input]
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[c, 1. - c]) for i, c in enumerate(closeness)])


def document(doc_id, name, agency, text):
    return {"doc_id": doc_id, "text": text,
            "metadata": {"사업명": name, "발주 기관": agency, "filename": f"{doc_id}.hwp"}}


QUESTION = "학사정보시스템 고도화 사업의 요구사항"
ENV_KEYS = ("RETRIEVAL_RERANK", "RETRIEVAL_CANDIDATES", "RETRIEVAL_HYBRID", "RETRIEVAL_HYBRID_VECTOR_K",
            "RETRIEVAL_HYBRID_BM25_K", "RETRIEVAL_RRF_K", "RETRIEVAL_REWRITE", "RETRIEVAL_REWRITE_MODEL")
# 연주님 추천 설정. max_per_doc은 run.py처럼 인자로 넘깁니다.
RECOMMENDED = {"RETRIEVAL_HYBRID": "on", "RETRIEVAL_RERANK": "lexical", "RETRIEVAL_CANDIDATES": "100"}


class RecordingClient(FakeClient):
    """답변 생성 요청을 기록해 모델에 실제로 보낸 검색 자료를 확인합니다."""

    def __init__(self):
        super().__init__()
        self.requests = []
        self.responses = SimpleNamespace(create=self.respond)

    def respond(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output_text="요구사항은 다음과 같습니다. [1]")

    def sent_sources(self):
        return json.loads(self.requests[-1]["input"].split("검색 자료(JSON):\n", 1)[1])


class RecommendedSettingsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        # 문서 2개 × 청크 3개. 문서당 상한 2면 top_k=5여도 최대 4개만 남습니다.
        build_index([
            document("bus", "버스정보시스템 구축", "가", "가 버스 노선 안내와 제안 안내"),
            document("bus", "버스정보시스템 구축", "가", "나 버스 정류장 정보와 제출 서류"),
            document("bus", "버스정보시스템 구축", "가", "다 버스 운행 관리 요구사항"),
            document("haksa", "학사정보시스템 고도화", "나", "가 학사정보시스템 고도화 요구사항 정의"),
            document("haksa", "학사정보시스템 고도화", "나", "나 학사 일정과 제안 안내"),
            document("haksa", "학사정보시스템 고도화", "나", "기타 학사 성적 관리 기능"),
        ], FakeClient(), cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        environment = {key: value for key, value in os.environ.items() if key not in ENV_KEYS}
        patcher = patch.dict(os.environ, {**environment, **RECOMMENDED}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = RecordingClient()

    def search(self, top_k=5, filters=None):
        return retrieve(QUESTION, self.client, self.index, self.chunks, self.config, top_k, filters, max_per_doc=2)

    def test_fewer_than_top_k_chunks_in_rerank_order(self):
        hits = self.search()
        self.assertEqual(len(hits), 4)  # top_k=5보다 적게 들어옴
        doc_ids = [hit["doc_id"] for hit in hits]
        self.assertTrue(all(doc_ids.count(doc_id) <= 2 for doc_id in doc_ids))
        rerank_scores = [hit["rerank_score"] for hit in hits]
        self.assertEqual(rerank_scores, sorted(rerank_scores, reverse=True))
        self.assertTrue(all(isinstance(hit["rrf_score"], float) for hit in hits))  # 하이브리드가 실제로 켜짐

    def test_generation_keeps_hit_count_and_order(self):
        hits = self.search()
        result = generate_answer(QUESTION, hits, self.client)
        self.assertTrue(result["answer"])
        self.assertEqual(len(self.client.requests), 1)
        # 결과에 저장된 sources와 모델에 보낸 검색 자료 모두 검색 순서 그대로, 번호는 1부터 연속입니다.
        for sources in (result["sources"], self.client.sent_sources()):
            self.assertEqual([source["citation"] for source in sources], list(range(1, len(hits) + 1)))
            self.assertEqual([source["text"] for source in sources], [hit["text"] for hit in hits])
        self.assertEqual([(s["chunk_id"], s["rerank_score"]) for s in result["sources"]],
                         [(hit["chunk_id"], hit["rerank_score"]) for hit in hits])

    def test_single_hit_is_answered(self):
        hits = self.search(top_k=1)
        result = generate_answer(QUESTION, hits, self.client)
        self.assertEqual([source["citation"] for source in result["sources"]], [1])
        self.assertEqual(len(self.client.sent_sources()), 1)

    def test_no_hits_skips_generation_call(self):
        hits = self.search(filters={"발주 기관": "없음"})
        self.assertEqual(hits, [])
        result = generate_answer(QUESTION, hits, self.client)
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["answer"])
        self.assertEqual(self.client.requests, [])


class ContextTests(unittest.TestCase):
    """모델에 보내는 검색 자료: 필요한 필드만, 사업 정보는 문서마다 한 번만."""

    def hit(self, doc_id, number, text, **extra):
        metadata = {"사업명": f"{doc_id} 사업", "발주 기관": f"{doc_id} 기관", "사업 금액": "1000",
                    "원문 계약 방법": "", "filename": f"{doc_id}.hwp", "source": f"files/{doc_id}.hwp",
                    "start_char": 0, "end_char": len(text), "section_path": "", "tables": [], **extra}
        return {"chunk_id": f"{doc_id}:{number}", "doc_id": doc_id, "text": text, "metadata": metadata,
                "score": .5, "rrf_score": .01, "rerank_score": 1.}

    def sent(self, hits):
        client = RecordingClient()
        result = generate_answer("질문", hits, client)
        return result, client.sent_sources()

    def test_scores_ids_and_positions_are_not_sent(self):
        hits = [self.hit("a", 0, "본문1"), self.hit("b", 0, "본문2")]
        result, sent = self.sent(hits)
        for item in sent:
            for key in ("score", "rrf_score", "rerank_score", "chunk_id", "doc_id", "metadata",
                        "filename", "source", "start_char", "end_char"):
                self.assertNotIn(key, item)
        # 결과의 sources는 평가가 읽으므로 전체 검색 결과 그대로입니다.
        self.assertEqual(result["sources"], [{"citation": i, **hit} for i, hit in enumerate(hits, 1)])

    def test_project_info_only_once_per_document_but_name_every_time(self):
        _, sent = self.sent([self.hit("a", 0, "본문1"), self.hit("a", 1, "본문2"), self.hit("b", 0, "본문3")])
        self.assertEqual([item["사업명"] for item in sent], ["a 사업", "a 사업", "b 사업"])
        self.assertEqual([item["발주 기관"] for item in sent], ["a 기관", "a 기관", "b 기관"])
        self.assertEqual([item.get("사업 정보") for item in sent], [{"사업 금액": "1000"}, None, {"사업 금액": "1000"}])

    def test_section_and_table_header_are_kept(self):
        table = {"table_id": "T1", "start_char": 0, "end_char": 9, "header_text": "| 항목 | 금액 |",
                 "header_start_char": 0, "header_end_char": 9}
        _, sent = self.sent([self.hit("a", 0, "| 예산 | 100 |", section_path="Ⅱ 사업 개요", tables=[table]),
                             self.hit("a", 1, "본문", tables=[{**table, "header_text": ""}])])
        self.assertEqual(sent[0]["section_path"], "Ⅱ 사업 개요")
        self.assertEqual(sent[0]["tables"], [{"table_id": "T1", "header_text": "| 항목 | 금액 |"}])
        self.assertNotIn("section_path", sent[1])  # 빈 값은 보내지 않음
        self.assertNotIn("tables", sent[1])

    def test_instructions_cover_format_and_missing_evidence_rules(self):
        client = RecordingClient()
        generate_answer("질문", [self.hit("a", 0, "본문")], client)
        instructions = client.requests[-1]["instructions"]
        for phrase in ("존댓말", "결론을 먼저", "마크다운 표 하나", "이름표를 붙이지 말고", "따로 한 줄로 쓰지도",
                       "답변 전체에서 한 번만", "사업명을 밝히세요", "되물으세요",
                       "계약 방법과 제출 방법", "반각 대괄호", "출처 번호를 붙이지 마세요"):
            self.assertIn(phrase, instructions)
        # 정해진 문구를 그대로 쓰라고 하면 같은 문장이 반복되어(10/7 Q2) 고정 문구는 넣지 않습니다.
        self.assertNotIn("'제공된 자료에서 확인할 수 없습니다'", instructions)


class FailureTests(unittest.TestCase):
    """빈 답변·잘린 답변·API 오류에도 멈추지 않고 안내 문구와 status를 돌려주는지 확인합니다."""

    HITS = [{"chunk_id": "a:0", "doc_id": "a", "text": "예산 100원", "metadata": {"사업명": "a 사업"}, "score": 1.}]

    def answer(self, respond):
        client = SimpleNamespace(responses=SimpleNamespace(create=respond))
        return generate_answer("예산은?", self.HITS, client)

    @staticmethod
    def request():
        return httpx.Request("POST", "https://api.openai.com/v1/responses")

    def test_ok_and_no_hits_status(self):
        result = self.answer(lambda **kw: SimpleNamespace(output_text=" 100원입니다. [1] ", status="completed"))
        self.assertEqual((result["answer"], result["status"]), ("100원입니다. [1]", "ok"))
        self.assertNotIn("error", result)
        self.assertEqual(generate_answer("예산은?", [], None)["status"], "no_hits")

    def test_citation_brackets_are_normalized(self):
        result = self.answer(lambda **kw: SimpleNamespace(output_text="예산은 100원입니다.【1】【 2 】 기간은 3개월［3］〔4〕 [5]"))
        self.assertEqual(result["answer"], "예산은 100원입니다.[1][2] 기간은 3개월[3][4] [5]")
        # 숫자가 아닌 괄호 내용은 바꾸지 않습니다.
        self.assertEqual(self.answer(lambda **kw: SimpleNamespace(output_text="【참고】 없음"))["answer"], "【참고】 없음")

    def test_empty_answer_returns_message_instead_of_error(self):
        for response in (SimpleNamespace(output_text=""), SimpleNamespace(output_text=None), SimpleNamespace(),
                         SimpleNamespace(output_text="", status="incomplete",
                                         incomplete_details=SimpleNamespace(reason="max_output_tokens"))):
            with self.subTest(response=response):
                result = self.answer(lambda **kw: response)
                self.assertEqual(result["status"], "empty")
                self.assertTrue(result["answer"])
                self.assertEqual(len(result["sources"]), 1)  # 실패해도 sources는 그대로 (평가가 읽음)

    def test_truncated_answer_keeps_text_and_adds_note(self):
        result = self.answer(lambda **kw: SimpleNamespace(
            output_text="예산은 100원이며", status="incomplete",
            incomplete_details=SimpleNamespace(reason="max_output_tokens")))
        self.assertEqual(result["status"], "truncated")
        self.assertTrue(result["answer"].startswith("예산은 100원이며"))
        self.assertIn("잘렸습니다", result["answer"])

    def test_timeout_connection_and_server_errors_do_not_stop(self):
        errors = [openai.APITimeoutError(request=self.request()),
                  openai.APIConnectionError(request=self.request()),
                  openai.InternalServerError("서버 오류", response=httpx.Response(500, request=self.request()),
                                             body=None)]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                def respond(**kw):
                    raise error
                result = self.answer(respond)
                self.assertEqual((result["status"], result["error"]), ("api_error", type(error).__name__))
                self.assertTrue(result["answer"])
                self.assertEqual(len(result["sources"]), 1)

    def test_rate_limit_still_reaches_run_py(self):
        def respond(**kw):
            raise openai.RateLimitError("한도", response=httpx.Response(429, request=self.request()), body=None)
        with self.assertRaises(openai.RateLimitError):
            self.answer(respond)


if __name__ == "__main__":
    unittest.main()
