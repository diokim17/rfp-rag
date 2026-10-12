"""박단비: 시나리오 B 생성 함수를 외부 API 없이 검증합니다 (입력 순서·입력 정리·status)."""

import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embedding import build_index, load_index
from generation import generate_answer
from retrieval import retrieve

QUESTION = "학사정보시스템 고도화 사업의 요구사항"


class FakeClient:
    """임베딩: 본문 첫 글자로 코사인 순서를 정함(가 > 나 > 다). 답변: 요청을 기록하고 정해진 답을 돌려줌."""

    CLOSENESS = {"학": 1., "가": .9, "나": .8, "다": .7}

    def __init__(self, answer="요구사항입니다.【1】"):
        self.requests = []
        self.embeddings = SimpleNamespace(create=lambda model, input: SimpleNamespace(data=[
            SimpleNamespace(index=i, embedding=[self.CLOSENESS.get(t[0], 0.), 1. - self.CLOSENESS.get(t[0], 0.)])
            for i, t in enumerate(input)]))
        self.responses = SimpleNamespace(create=lambda **kw: self.requests.append(kw) or SimpleNamespace(output_text=answer))

    def sent(self):
        """모델에 실제로 보낸 검색 자료."""
        return json.loads(self.requests[-1]["input"].split("검색 자료(JSON):\n", 1)[1])


def hit(doc_id, text, **meta):
    return {"chunk_id": f"{doc_id}:0", "doc_id": doc_id, "text": text, "score": .5, "rrf_score": .01, "rerank_score": 1.,
            "metadata": {"사업명": f"{doc_id} 사업", "발주 기관": f"{doc_id} 기관", "filename": f"{doc_id}.hwp",
                         "start_char": 0, "end_char": len(text), **meta}}


class RecommendedSettingsTests(unittest.TestCase):
    """[연주님 요청] 하이브리드 + lexical + 문서당 상한 2에서 청크가 5개 미만·리랭킹 순서로 들어와도 정상 동작."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        doc = lambda doc_id, name, text: {"doc_id": doc_id, "text": text, "metadata": {"사업명": name, "filename": doc_id}}
        # 문서 2개 × 청크 3개 → 상한 2면 top_k=5여도 4개만 남음
        build_index([doc("bus", "버스", "가 버스 노선 안내"), doc("bus", "버스", "나 버스 정류장 정보"),
                     doc("bus", "버스", "다 버스 운행 요구사항"), doc("haksa", "학사", "가 학사정보시스템 고도화 요구사항"),
                     doc("haksa", "학사", "나 학사 일정 안내"), doc("haksa", "학사", "기타 학사 성적 관리")],
                    FakeClient(), cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def search(self, client, filters=None):
        with patch.dict(os.environ, {"RETRIEVAL_HYBRID": "on", "RETRIEVAL_RERANK": "lexical", "RETRIEVAL_CANDIDATES": "100"}):
            return retrieve(QUESTION, client, self.index, self.chunks, self.config, 5, filters, max_per_doc=2)

    def test_fewer_chunks_keep_rerank_order_into_generation(self):
        client = FakeClient()
        hits = self.search(client)
        self.assertEqual(len(hits), 4)
        self.assertTrue(all([h["doc_id"] for h in hits].count(d) <= 2 for d in ("bus", "haksa")))
        self.assertEqual([h["rerank_score"] for h in hits], sorted((h["rerank_score"] for h in hits), reverse=True))
        result = generate_answer(QUESTION, hits, client)
        for sources in (result["sources"], client.sent()):  # 결과와 모델 입력 모두 검색 순서 그대로, 번호 1부터
            self.assertEqual([s["citation"] for s in sources], [1, 2, 3, 4])
            self.assertEqual([s["text"] for s in sources], [h["text"] for h in hits])

    def test_no_hits_skips_api(self):
        client = FakeClient()
        result = generate_answer(QUESTION, self.search(client, {"발주 기관": "없음"}), client)
        self.assertEqual((result["status"], result["sources"], client.requests), ("no_hits", [], []))


class ContextTests(unittest.TestCase):
    """모델 입력 정리: 점수·ID·위치는 빼고, 사업 정보는 문서마다 한 번, 섹션·표 머리글은 유지."""

    def test_context_fields(self):
        table = {"table_id": "T1", "header_text": "| 항목 | 금액 |", "start_char": 0, "end_char": 9}
        hits = [hit("a", "본문1", **{"사업 금액": "1000", "section_path": "Ⅱ 개요", "tables": [table]}),
                hit("a", "본문2", **{"사업 금액": "1000"}), hit("b", "본문3", **{"사업 금액": "2000"})]
        client = FakeClient()
        result = generate_answer("질문", hits, client)
        sent = client.sent()
        for item in sent:
            self.assertFalse({"score", "rrf_score", "rerank_score", "chunk_id", "doc_id", "metadata"} & set(item))
        self.assertEqual([i["사업명"] for i in sent], ["a 사업", "a 사업", "b 사업"])
        self.assertEqual([i.get("사업 정보") for i in sent], [{"사업 금액": "1000"}, None, {"사업 금액": "2000"}])
        self.assertEqual((sent[0]["section_path"], sent[0]["tables"]), ("Ⅱ 개요", [{"table_id": "T1", "header_text": "| 항목 | 금액 |"}]))
        self.assertNotIn("section_path", sent[1])
        self.assertEqual(result["sources"], [{"citation": i, **h} for i, h in enumerate(hits, 1)])  # 결과는 전체 그대로


class StatusTests(unittest.TestCase):
    """정상 답변은 출처 괄호를 [n]으로 바꾸고, 빈 답변·잘린 답변은 안내 문구와 status를 돌려줌."""

    def answer(self, response):
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: response))
        return generate_answer("예산은?", [hit("a", "예산 100원")], client)

    def test_ok_empty_truncated(self):
        ok = self.answer(SimpleNamespace(output_text=" 100원입니다.【1】 "))
        self.assertEqual((ok["answer"], ok["status"]), ("100원입니다.[1]", "ok"))
        self.assertEqual(self.answer(SimpleNamespace(output_text=""))["status"], "empty")
        cut = self.answer(SimpleNamespace(output_text="100원이며", status="incomplete",
                                          incomplete_details=SimpleNamespace(reason="max_output_tokens")))
        self.assertEqual(cut["status"], "truncated")
        self.assertIn("잘렸습니다", cut["answer"])


if __name__ == "__main__":
    unittest.main()
