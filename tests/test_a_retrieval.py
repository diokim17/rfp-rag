"""시나리오 A 검색(a_retrieval.py)의 A 전용 기본값과 옵션을 외부 API·모델 없이 검증합니다.

- 범위 밖 차단 기준(min_score)은 A 임베딩으로 잰 0.53이 기본입니다.
- rerank_section(기본 켬)은 재정렬 입력에만 절 제목을 붙입니다.
가짜 임베딩·문서는 test_retrieval_rerank.py의 것을 같이 씁니다.
"""

import os
import tempfile
import unittest
from unittest.mock import patch

from a_embedding import build_index, load_index
import a_retrieval
from a_retrieval import retrieval_options, retrieve
from test_retrieval_rerank import FakeClient, document

QUESTION = "학사정보시스템 고도화 사업의 요구사항"
# 하이브리드·상한·후보 확장을 꺼서 재정렬 순서만 보이게 합니다.
PLAIN = {"hybrid": False, "max_per_doc": "none", "expand": 0}


class ARetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        build_index([
            document("bus", "버스정보시스템 구축", "가", "가 요구사항 목록과 제안 안내"),
            document("haksa", "학사정보시스템 고도화", "나", "다 요구사항 목록과 제안 안내"),
            document("water", "용수공급 타당성조사", "나", "나 요구사항 목록과 제안 안내"),
            # 범위 밖 질문(가짜 임베딩 코사인 0)과 구분되도록 모든 본문이 RFP 쪽 벡터를 갖게 합니다.
            document("etc", "학사정보시스템 고도화 감리", "다", "가 일정과 예산"),
        ], cls.client, cls.directory.name, model="test-embedding")
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        # 실행하는 사람의 RETRIEVAL_* 설정이 결과를 바꾸지 않게 지우고, 리랭커는 설치된 것으로 칩니다.
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("RETRIEVAL_")}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        for name, value in (("_cross_encoder_available", True), ("rerank_model_source", "fake-reranker")):
            patcher = patch.object(a_retrieval, name, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def search(self, question=QUESTION, chunks=None, **options):
        return retrieve(question, self.client, self.index, chunks or self.chunks, self.config, 4,
                        **{**PLAIN, "rerank": "lexical", **options})

    def test_min_score_default_is_a_value_and_settable(self):
        """A는 A 임베딩으로 잰 0.53이 기본이고, 환경 변수·인자로 바꾸거나 끌 수 있습니다."""
        self.assertEqual(retrieval_options()["min_score"], 0.53)
        self.assertEqual(self.search(question="점심 메뉴 추천해줘"), [])
        self.assertEqual(len(self.search()), 4)  # RFP 질문은 막지 않음
        self.assertEqual(retrieval_options(min_score=0.3)["min_score"], 0.3)
        with patch.dict(os.environ, {"RETRIEVAL_MIN_SCORE": "off"}):
            self.assertIsNone(retrieval_options()["min_score"])
            self.assertEqual(len(self.search(question="점심 메뉴 추천해줘")), 4)
        # B 환경 변수(RFP_SCENARIO=B)가 남아 있어도 A 기본값을 씁니다.
        with patch.dict(os.environ, {"RFP_SCENARIO": "B"}):
            self.assertEqual(retrieval_options()["min_score"], 0.53)

    def test_rerank_section_adds_section_title_to_rerank_input_only(self):
        # 같은 문서 안에서 절 제목으로만 구분되는 상황: 절 제목에 '발표'가 든 청크만 점수를 받습니다.
        chunks = [{**chunk, "metadata": {**chunk["metadata"], "section_path": section}}
                  for chunk, section in zip(self.chunks, ["Ⅰ 개요", "Ⅳ 제안 > 3 발표", "", "Ⅱ 일정"])]
        seen = []

        def scorer(question, passages, model_name):
            seen.append(passages)
            return [3. if "발표" in passage else 0. for passage in passages]

        search = lambda **options: self.search(chunks=chunks, rerank="cross-encoder", **options)
        with patch.object(a_retrieval, "_cross_encoder_scores", scorer):
            off = search(rerank_section=False)
            on = search()  # 기본은 켬
            with patch.dict(os.environ, {"RETRIEVAL_RERANK_SECTION": "off"}):
                self.assertEqual(search(), off)
                self.assertEqual(search(rerank_section=True), on)  # 인자가 환경 변수보다 우선
        self.assertFalse(any("발표" in passage for passage in seen[0]))  # 끄면 절 제목을 넣지 않음
        self.assertEqual(on[0]["chunk_id"], chunks[1]["chunk_id"])
        self.assertIn(f"학사정보시스템 고도화 나\nⅣ 제안 > 3 발표\n{chunks[1]['text']}", seen[1])
        # 절 제목이 빈 청크는 기존 입력과 같습니다.
        self.assertEqual(sum(passage.count("\n") == 1 for passage in seen[1]), 1)
        # 반환 청크(본문 포함)와 score는 바뀌지 않고 순서만 달라집니다.
        strip = lambda hits: {hit["chunk_id"]: {k: v for k, v in hit.items() if k != "rerank_score"} for hit in hits}
        self.assertEqual(strip(on), strip(off))
        with self.assertRaises(ValueError):
            search(rerank_section="maybe")
        self.assertIsNone(retrieval_options(rerank="none", rerank_section=True)["rerank_section"])

    def test_retrieval_signals(self):
        hit = lambda doc_id, score, rerank=None: {"doc_id": doc_id, "score": score,
                                                  **({} if rerank is None else {"rerank_score": rerank})}
        # 1위 문서 a가 2자리, 재정렬 격차는 1위(5) - a가 아닌 문서 중 최고(2)
        self.assertEqual(a_retrieval.retrieval_signals([hit("a", .6, 5.), hit("b", .7, 2.), hit("a", .5, 1.), hit("c", .4, 0.)]),
                         {"docs": 3, "top_doc_share": .5, "best_score": .7, "rerank_gap": 3.})
        # 결과 없음(범위 밖 차단·필터 0건)
        self.assertEqual(a_retrieval.retrieval_signals([]),
                         {"docs": 0, "top_doc_share": None, "best_score": None, "rerank_gap": None})
        # 재정렬을 안 했거나 한 문서뿐이면 격차는 None
        self.assertIsNone(a_retrieval.retrieval_signals([hit("a", .6), hit("b", .5)])["rerank_gap"])
        self.assertEqual(a_retrieval.retrieval_signals([hit("a", .6, 3.), hit("a", .5, 1.)]),
                         {"docs": 1, "top_doc_share": 1., "best_score": .6, "rerank_gap": None})


class AUncapNamedTests(unittest.TestCase):
    """uncap_named: 질문이 사업 하나를 이름으로 가리키면 그 질문만 문서당 상한을 풉니다(기본 끔)."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.client = FakeClient()
        # 10자 청크: 학사 사업은 청크 3개, 나머지는 1~2개. 사업명이 서로 겹치지 않아 고유 2-gram으로 구분됩니다.
        build_index([
            document("haksa", "대전대학교 학사행정 고도화", "가", "가" * 10 + "나" * 10 + "다" * 10),
            document("bus", "버스운행 관리 구축", "나", "가" * 10 + "나" * 10),
            document("water", "용수공급 타당성조사", "다", "가" * 10),
            document("library", "도서관 전산화", "라", "나" * 10),
        ], cls.client, cls.directory.name, model="test-embedding", chunk_size=10, chunk_overlap=0)
        cls.index, cls.chunks, cls.config = load_index(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith("RETRIEVAL_")}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def docs(self, question, **options):
        # 가짜 임베딩은 사업명을 반영하지 않아 코사인이 낮으므로 범위 밖 차단은 끕니다.
        hits = retrieve(question, self.client, self.index, self.chunks, self.config, 4,
                        **{**PLAIN, "rerank": "lexical", "max_per_doc": 1, "min_score": "none", **options})
        return [hit["doc_id"] for hit in hits]

    def test_single_named_project_lifts_cap_only_when_on(self):
        question = "대전대학교 학사행정 고도화 사업의 일정"
        self.assertFalse(retrieval_options()["uncap_named"])  # 기본 끔
        self.assertEqual(self.docs(question).count("haksa"), 1)  # 끄면 상한 1 그대로
        on = self.docs(question, uncap_named=True)
        self.assertEqual(on.count("haksa"), 3)  # 그 사업 청크 3개가 모두 들어옴
        self.assertEqual(len(on), 4)
        self.assertEqual(self.docs(question, uncap_named=True, hybrid=True).count("haksa"), 3)  # 하이브리드 경로도 같음
        with patch.dict(os.environ, {"RETRIEVAL_UNCAP_NAMED": "on"}):
            self.assertEqual(self.docs(question), on)
            self.assertEqual(self.docs(question, uncap_named=False).count("haksa"), 1)  # 인자가 환경 변수보다 우선

    def test_unnamed_or_two_named_questions_keep_cap(self):
        # 사업명이 없는 질문, 두 사업을 가리키는 비교 질문, 사업명 한 단어만 겹치는 질문은 상한을 풀지 않습니다.
        for question in ("사업의 일정 알려줘", "대전대학교 학사행정 고도화와 버스운행 관리 구축 비교", "학사 일정"):
            self.assertEqual(self.docs(question, uncap_named=True), self.docs(question), question)
        self.assertIsNone(retrieval_options(rerank="none", uncap_named=True)["uncap_named"])
        with self.assertRaises(ValueError):
            retrieval_options(uncap_named="maybe")


if __name__ == "__main__":
    unittest.main()
