"""외부 API 없이 평가 채점·집계·판정 캐시를 검증합니다."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))  # discover 외 실행에서도 테스트 보조 모듈을 찾음

from a_evaluation import amounts, citations, evaluate, is_refusal, make_llm_judge, periods
from a_observability import Trace, model_call
from test_a_contract_observability import FakeLangfuse


def source(citation, doc_id, text=""):
    return {"citation": citation, "doc_id": doc_id, "text": text, "metadata": {}, "chunk_id": f"{doc_id}:0", "score": .5}


def answer_with(answer, *docs, texts=None, status="ok"):
    texts = texts or [""] * len(docs)
    return {"question": "q", "answer": answer, "sources": [source(i, d, t) for i, (d, t) in enumerate(zip(docs, texts), 1)],
            "model": "fake", "status": status}


class NumericNormalizationTests(unittest.TestCase):
    def test_amount_notations(self):
        for text in ("470백만원", "4억 7천만 원", "470,000,000원", "470,000천원", "4.7억 원", "사억칠천만원", "￦470,000,000"):
            with self.subTest(text=text):
                self.assertIn(470_000_000, amounts(text))
        self.assertIn(195_030_000, amounts("일금 일억구천오백삼만원정"))
        self.assertEqual(amounts("5억이 아니라"), {500_000_000})
        self.assertEqual(amounts("2억 사업과 3천만 원"), {200_000_000, 30_000_000})
        self.assertEqual(amounts("3천만 원"), {30_000_000})
        for text in ("계약일로부터 3개월", "만족도 조사", "PC 740대", "발표 20분", "이 원칙"):
            with self.subTest(text=text):
                self.assertEqual(amounts(text), set())

    def test_periods_and_dates(self):
        self.assertIn(("개월", 3), periods("계약일로부터 3 개월"))
        for text in ("2024. 12. 31.", "2024년 12월 31일", "‘24년 12월 31일까지"):
            with self.subTest(text=text):
                self.assertIn(("날짜", "2024-12-31"), periods(text))
        self.assertNotIn(("일", 31), periods("2024년 12월 31일"))

    def test_citations_and_refusal(self):
        self.assertEqual(citations("a [1] b [2][4] c [1, 3] d [5-6]"), {1, 2, 3, 4, 5, 6})
        self.assertTrue(is_refusal("제공된 자료로는 확인할 수 없습니다."))
        self.assertFalse(is_refusal("사업 예산은 1억 원입니다 [1]"))


class BackwardCompatibilityTests(unittest.TestCase):
    def test_legacy_cases_keep_existing_summary(self):
        result = evaluate([{"question": "예산", "expected_doc_ids": ["a", "b"], "expected_keywords": ["100원"]}],
                          lambda q, f: answer_with("예산은 100원입니다", "a"))
        summary = result["summary"]
        self.assertEqual((summary["count"], summary["recall_at_k"], summary["keyword_coverage"]), (1, .5, 1.))
        self.assertEqual(summary["answer_accuracy"], 1.)  # grading이 없으면 기대 키워드로 채점
        no_keywords = evaluate([{"question": "예산", "expected_doc_ids": ["a"]}], lambda q, f: answer_with("답", "a"))
        self.assertIsNone(no_keywords["summary"]["answer_accuracy"])  # 키워드도 없으면 채점하지 않음
        self.assertTrue(all(isinstance(v, (int, float)) or v is None for v in summary.values()))
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_invalid_cases_fail_before_answering(self):
        bad = [{"question": "q", "expected_doc_ids": []},
               {"question": "q", "expected_doc_ids": ["a"], "grading": "unknown"},
               {"question": "q", "expected_doc_ids": ["a"], "grading": "numeric"},
               {"question": "q", "answerable": "no"},
               {"question": "q", "answerable": False, "expected_doc_ids": "a"},
               {"question": "q", "answerable": False, "expected_doc_ids": [""]}]
        for case in bad:
            with self.subTest(case=case), self.assertRaises(ValueError):
                evaluate([case], lambda q, f: self.fail("검증 전에 답변하면 안 됨"))


class GradingTests(unittest.TestCase):
    def run_one(self, case, answer, judge=None):
        return evaluate([{"question": "q", **case}], lambda q, f: answer, judge=judge)

    def test_numeric_accepts_notation_and_requires_gold_citation(self):
        case = {"expected_doc_ids": ["a"], "grading": "numeric", "expected_values": {"amount_won": [470_000_000]},
                "requires_gold_citation": True}
        self.assertEqual(self.run_one(case, answer_with("470백만원입니다 [1]", "a"))["summary"]["answer_accuracy"], 1.)
        self.assertEqual(self.run_one(case, answer_with("4억 8천만 원입니다 [1]", "a"))["summary"]["answer_accuracy"], 0.)
        self.assertEqual(self.run_one(case, answer_with("4억 7천만 원입니다", "a"))["summary"]["answer_accuracy"], 0.)
        wrong_doc = self.run_one(case, answer_with("4억 7천만 원입니다 [2]", "a", "b"))
        self.assertEqual(wrong_doc["summary"]["answer_accuracy"], 0.)

    def test_amount_and_period_both_required(self):
        case = {"expected_doc_ids": ["a"], "grading": "numeric",
                "expected_values": {"amount_won": [150_000_000], "period": [["일", 180]]}}
        self.assertEqual(self.run_one(case, answer_with("180일, 1억 5천만 원", "a"))["summary"]["answer_accuracy"], 1.)
        self.assertEqual(self.run_one(case, answer_with("1억 5천만 원", "a"))["summary"]["answer_accuracy"], 0.)

    def test_keyword_needs_all_and_ignores_spaces(self):
        case = {"expected_doc_ids": ["a"], "grading": "keyword", "expected_keywords": ["시스템 전체", "12개월"]}
        self.assertEqual(self.run_one(case, answer_with("시스템전체, 12 개월", "a"))["summary"]["answer_accuracy"], 1.)
        result = self.run_one(case, answer_with("시스템 전체", "a"))
        self.assertEqual(result["summary"]["answer_accuracy"], 0.)
        self.assertEqual(result["summary"]["keyword_coverage"], .5)  # 기존 포함 비율은 그대로

    def test_llm_judge_optional(self):
        case = {"expected_doc_ids": ["a"], "grading": "llm", "requires_gold_citation": True}
        ungraded = self.run_one(case, answer_with("답 [1]", "a"))["summary"]
        self.assertEqual((ungraded["graded_count"], ungraded["ungraded_count"]), (0, 1))
        self.assertEqual(self.run_one(case, answer_with("답 [1]", "a"), lambda *a: "partial")["summary"]["answer_accuracy"], .5)
        self.assertEqual(self.run_one(case, answer_with("답", "a"), lambda *a: "correct")["summary"]["answer_accuracy"], 0.)

    def test_unanswerable_and_false_premise(self):
        off_topic = {"answerable": False, "grading": "refusal"}
        result = self.run_one(off_topic, answer_with("제공된 자료로는 확인할 수 없습니다.", "a"))
        self.assertEqual(result["summary"]["refusal_accuracy"], 1.)
        self.assertIsNone(result["summary"]["recall_at_k"])
        self.assertEqual(self.run_one(off_topic, answer_with("예산은 3억 원입니다 [1]", "a"))["summary"]["refusal_accuracy"], 0.)
        no_hits = self.run_one(off_topic, answer_with("조건에 맞는 문서가 없어 답변할 수 없습니다.", status="no_hits"))
        self.assertEqual(no_hits["summary"]["refusal_accuracy"], 1.)
        premise = {"expected_doc_ids": ["a"], "grading": "numeric", "expected_values": {"amount_won": [130_000_000]},
                   "forbidden_values": {"amount_won": [500_000_000]}}
        self.assertEqual(self.run_one(premise, answer_with("네, 5억 원입니다", "a"))["summary"]["answer_accuracy"], 0.)  # 전제에 동의
        self.assertEqual(self.run_one(premise, answer_with("5억이 아니라 1억 3천만 원입니다", "a"))["summary"]["answer_accuracy"], 1.)
        self.assertEqual(self.run_one(premise, answer_with("1억 3천만 원입니다", "a"))["summary"]["answer_accuracy"], 1.)


class TeamRequestTests(unittest.TestCase):
    """답 없는 질문 지원·지표 분리·키워드 숫자 정규화 요청 (답 있는 42 + 답 없는 8 = 50문항)."""

    def test_fifty_cases_with_separate_metrics(self):
        answerable = [{"id": f"a{i}", "question": f"예산 {i}", "expected_doc_ids": ["d"], "expected_keywords": ["130,000,000"]}
                      for i in range(42)]
        unanswerable = [{"id": f"u{i}", "question": f"없는 사업 {i}", "answerable": False} for i in range(8)]
        replies = {**{f"예산 {i}": answer_with("사업 예산은 130,000천원입니다 [1]" if i < 40 else "모르겠습니다", "d")
                      for i in range(42)},
                   **{f"없는 사업 {i}": (answer_with("조건에 맞는 문서가 없어 답변할 수 없습니다.", status="no_hits") if i < 4 else
                                       answer_with("해당 내용은 문서에서 확인되지 않습니다.", "d") if i < 6 else
                                       answer_with("사업 예산은 3억 원입니다 [1]", "d")) for i in range(8)}}
        result = evaluate(answerable + unanswerable, lambda q, f: replies[q])
        s = result["summary"]
        self.assertEqual((s["count"], s["answerable_count"], s["unanswerable_count"]), (50, 42, 8))
        self.assertEqual(s["recall_at_k"], 1.)  # 답 없는 8문항은 Recall에서 제외
        self.assertTrue(all(r["recall_at_k"] is None for r in result["records"][42:]))
        self.assertAlmostEqual(s["answer_accuracy"], 40 / 42)  # '130,000천원' = 키워드 '130,000,000'
        self.assertEqual(s["keyword_coverage"], 0.)  # 기존 포함률(정규화 없음)은 계약대로 유지
        self.assertEqual((s["refusal_accuracy"], s["refusal_graded_count"]), (.75, 8))  # no_hits 4 + '확인되지 않' 2
        self.assertAlmostEqual(s["unanswerable_hallucination_rate"], .25)

    def test_keyword_number_and_date_notation(self):
        cases = [("130,000,000", "130,000천원입니다"), ("130,000,000", "1억 3천만 원"), ("2024-12-31", "‘24년 12월 31일까지"),
                 ("2024-12-31", "2024. 12. 31."), ("5개월", "착수일로부터 5 개월"), ("시스템 전체", "시스템전체")]
        for keyword, answer in cases:
            with self.subTest(keyword=keyword, answer=answer):
                r = evaluate([{"question": "q", "expected_doc_ids": ["d"], "expected_keywords": [keyword]}],
                             lambda q, f: answer_with(answer, "d"))
                self.assertEqual(r["summary"]["answer_accuracy"], 1.)
        wrong = evaluate([{"question": "q", "expected_doc_ids": ["d"], "expected_keywords": ["130,000,000"]}],
                         lambda q, f: answer_with("140,000천원", "d"))
        self.assertEqual(wrong["summary"]["answer_accuracy"], 0.)


class ReviewFixTests(unittest.TestCase):
    """재검토에서 찾은 문제의 회귀 테스트."""

    def one(self, case, answer, judge=None):
        return evaluate([{"question": "q", **case}], lambda q, f: answer, judge=judge)

    def test_unit_less_numeric_keyword_still_matches_as_text(self):
        r = self.one({"expected_doc_ids": ["d"], "expected_keywords": ["100,000"]}, answer_with("과업비는 100,000천원입니다", "d"))
        self.assertEqual(r["summary"]["answer_accuracy"], 1.)

    def test_partial_refusal_with_invented_value_is_hallucination(self):
        case = {"answerable": False}
        mixed = self.one(case, answer_with("예산은 3억 원이며, 기간은 문서에서 확인되지 않습니다.", "d"))
        self.assertEqual(mixed["summary"]["refusal_accuracy"], 0.)
        self.assertEqual(self.one(case, answer_with("해당 사업은 문서에서 확인되지 않습니다.", "d"))["summary"]["refusal_accuracy"], 1.)

    def test_local_model_citation_formats(self):
        self.assertEqual(citations("[출처 1] 와 [문서 2], [3번], 【4】"), {1, 2, 3, 4})
        self.assertEqual(citations("(1) 사업 개요"), set())
        case = {"expected_doc_ids": ["d"], "expected_keywords": ["정답"], "requires_gold_citation": True}
        self.assertEqual(self.one(case, answer_with("정답 [출처 1]", "d"))["summary"]["answer_accuracy"], 1.)

    def test_content_accuracy_separates_citation_failures(self):
        case = {"expected_doc_ids": ["d"], "expected_keywords": ["정답"], "requires_gold_citation": True}
        s = self.one(case, answer_with("정답입니다", "d"))["summary"]  # 내용은 맞지만 인용 없음
        self.assertEqual((s["answer_accuracy"], s["answer_content_accuracy"]), (0., 1.))

    def test_cluster_defaults_to_gold_documents(self):
        cases = [{"question": f"q{i}", "expected_doc_ids": ["d" if i < 9 else f"e{i}"], "expected_keywords": ["정답"]}
                 for i in range(12)]
        answers = iter([answer_with("정답", "d")] * 9 + [answer_with("오답", "x")] * 3)
        s = evaluate(cases, lambda q, f: next(answers))["summary"]
        self.assertIsNotNone(s["answer_accuracy_dedup_ci95_low"])  # 클러스터 4개(d, e9, e10, e11)

    def test_corrected_false_premise_with_keyword_grading(self):
        case = {"expected_doc_ids": ["d"], "expected_keywords": ["130,000,000"], "forbidden_values": {"amount_won": [500_000_000]}}
        self.assertEqual(self.one(case, answer_with("5억이 아니라 1억 3천만 원입니다", "d"))["summary"]["answer_accuracy"], 1.)
        self.assertEqual(self.one(case, answer_with("네, 5억 원입니다", "d"))["summary"]["answer_accuracy"], 0.)

    def test_latency_without_no_hits_and_unknown_status(self):
        cases = [{"question": f"q{i}", "expected_doc_ids": ["d"]} for i in range(3)]
        answers = iter([answer_with("a", "d"), answer_with("b", status="no_hits"), answer_with("c", "d", status="loading")])
        s = evaluate(cases, lambda q, f: next(answers))["summary"]
        self.assertEqual((s["status_other_count"], s["status_no_hits_count"]), (1, 1))
        self.assertIsNotNone(s["latency_ok_mean_seconds"])

    def test_value_format_and_judge_output_validation(self):
        for values in ({"period": [["개월", "3"]]}, {"amount_won": ["130000000"]}, {"period": [["날짜", "2024.12.31"]]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate([{"question": "q", "expected_doc_ids": ["d"], "grading": "numeric", "expected_values": values}],
                         lambda q, f: self.fail())
        with self.assertRaises(ValueError):
            self.one({"expected_doc_ids": ["d"], "grading": "llm"}, answer_with("a [1]", "d"), judge=lambda *a: "maybe")

    def test_judge_tokens_are_kept_out_of_answer_cost(self):
        trace = Trace({}, FakeLangfuse())

        def judge(case, answer, sources):
            model_call("judge", "gpt-5-mini", lambda: SimpleNamespace(usage=SimpleNamespace(input_tokens=50, output_tokens=5)))
            return "correct"

        def answer_fn(question, filters):
            model_call("generation", "gpt-5-mini", lambda: SimpleNamespace(usage=SimpleNamespace(input_tokens=100, output_tokens=20)))
            return answer_with("답 [1]", "d")

        with trace.run("t"):
            r = evaluate([{"question": "q", "expected_doc_ids": ["d"], "grading": "llm"}], answer_fn, judge=judge)
        self.assertEqual(r["records"][0]["case_token_usage"], {"gpt-5-mini": {"input": 100, "output": 20}})
        self.assertEqual(r["records"][0]["judge_token_usage"], {"gpt-5-mini": {"input": 50, "output": 5}})
        self.assertEqual((r["summary"]["input_tokens_per_case"], r["summary"]["judge_input_tokens"]), (100, 50))


class PullRequestReviewTests(unittest.TestCase):
    """PR #20 리뷰: eval_v4 u05~u07(관련 사업은 있지만 RFP에 없는 정보) 형식, PR #18 범위 밖 안내 문구."""

    def test_unanswerable_case_may_list_related_documents(self):
        case = {"question": "낙찰 업체는?", "answerable": False, "expected_doc_ids": ["d"]}
        result = evaluate([case], lambda q, f: answer_with("RFP에서는 낙찰 업체가 확인되지 않습니다.", "d"))
        s = result["summary"]
        self.assertEqual((s["unanswerable_count"], s["refusal_accuracy"]), (1, 1.))
        self.assertIsNone(s["recall_at_k"])  # 답 없는 문항은 Recall에서 제외
        self.assertIsNone(result["records"][0]["gold_cited"])

    def test_off_topic_and_new_no_hits_phrases_are_refusals(self):
        off_topic = "RFP 문서에 관한 질문에만 답할 수 있습니다. 사업명이나 궁금한 항목을 넣어 다시 질문해 주세요."
        no_hits = "질문과 관련된 RFP 문서가 없어 답변할 수 없습니다."
        for answer in (off_topic, no_hits):
            with self.subTest(answer=answer):
                self.assertTrue(is_refusal(answer))
                r = evaluate([{"question": "오늘 날씨는?", "answerable": False}], lambda q, f: answer_with(answer))
                self.assertEqual(r["summary"]["refusal_accuracy"], 1.)


class RefusalRuleTests(unittest.TestCase):
    """법령 조항 번호, 끝의 단서 문장, 답 없는 문항에서 덧붙인 실제 값의 판정."""

    def test_article_numbers_are_not_amounts(self):
        self.assertEqual(amounts("소프트웨어 진흥법 제50조, 시행령 제47조 제1항"), set())
        self.assertEqual(amounts("제12조의2에 따라 사업비는 1억 원"), {100_000_000})
        self.assertEqual(amounts("5조 원 규모"), {5_000_000_000_000})  # '제'가 없는 실제 금액은 유지

    def test_not_specified_phrases_are_refusals(self):
        """'명시되지 않았습니다'·'기재되어 있지 않습니다'도 거절 (A u03·u06·u07, B shn-0044 u05가 이 표현으로 환각 처리됐음)."""
        for answer in ("2세대 전자조달시스템 기반구축사업: 제공된 문서들에서는 최종 낙찰 업체가 기재되어 있지 않습니다.",
                       "해당 사업의 낙찰 업체는 문서에 명시되지 않았습니다."):
            with self.subTest(answer=answer):
                self.assertTrue(is_refusal(answer))
        self.assertFalse(is_refusal("예산은 1억 원이며 기간은 명시되지 않았습니다 [1]."))  # 값을 답한 첫 문장은 거절 아님

    def test_trailing_caveat_is_not_a_refusal(self):
        for answer in ("사업 범위는 학사 시스템 고도화입니다 [1]. 단, 세부 일정은 문서에서 확인되지 않습니다.",
                       "사업 범위는 학사 시스템 고도화입니다[1][2]. 다만 세부 일정은 확인되지 않습니다.",
                       "기간은 2024. 12. 31.까지이며 그 밖의 내용은 확인되지 않습니다."):
            with self.subTest(answer=answer):
                self.assertFalse(is_refusal(answer))
        self.assertTrue(is_refusal("해당 내용은 문서에서 확인되지 않습니다. 다만 유사 사업이 있습니다 [1]."))
        result = evaluate([{"question": "q", "expected_doc_ids": ["d"], "expected_keywords": ["고도화"]}],
                          lambda q, f: answer_with("사업 범위는 고도화입니다 [1]. 단, 세부 일정은 확인되지 않습니다.", "d"))
        self.assertEqual(result["summary"]["false_refusal_rate"], 0.)

    def test_unanswerable_allows_only_related_document_values(self):
        sources = [source(1, "d", "사업예산: 130,000,000원(VAT 포함)"), source(2, "x", "사업예산: 200,000,000원")]
        reply = lambda text: {"question": "q", "answer": text, "sources": sources, "model": "fake", "status": "ok"}
        related = {"question": "낙찰 업체는?", "answerable": False, "expected_doc_ids": ["d"]}
        unrelated = {"question": "없는 사업 예산은?", "answerable": False}
        cases = [(related, "낙찰 업체는 확인되지 않습니다. 참고로 이 사업의 예산은 130,000,000원입니다 [1].", 1.),
                 (related, "낙찰 업체는 확인되지 않습니다. 참고로 사업 예산은 200,000,000원입니다 [2].", 0.),  # 다른 사업 값
                 (related, "낙찰 업체는 확인되지 않습니다. 참고로 사업 예산은 777,000,000원입니다 [1].", 0.),  # 근거에 없는 값
                 # 정답 문서가 없는 문항: 답변이 인용한 출처에 실제로 있는 값은 지어낸 값이 아님
                 (unrelated, "해당 사업은 확인되지 않습니다. 참고로 사업 예산은 200,000,000원입니다 [2].", 1.),
                 (unrelated, "해당 사업은 확인되지 않습니다. 참고로 사업 예산은 200,000,000원입니다 [1].", 0.),  # 인용한 [1]에 없는 값
                 (unrelated, "해당 사업은 확인되지 않습니다. 참고로 사업 예산은 777,000,000원입니다 [2].", 0.),  # 어느 출처에도 없는 값
                 (unrelated, "해당 사업은 확인되지 않습니다. 참고로 사업 예산은 200,000,000원입니다.", 1.),  # 인용 없음 → 검색된 출처 기준
                 (unrelated, "해당 사업은 문서에서 확인되지 않습니다.", 1.)]
        for case, text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(evaluate([case], lambda q, f: reply(text))["summary"]["refusal_accuracy"], expected)


class ClarifyAndCitedValueTests(unittest.TestCase):
    """u10(대상을 되묻는 답)과 u09(정답 문서가 없는 문항에서 출처의 날짜 인용)."""

    def test_clarifying_question_counts_as_refusal(self):
        u10 = "어떤 사업에 대한 질문인지 특정되지 않아 어떤 사업을 말씀하시는지 알려주십시오.\n제공된 자료로는 확인할 수 없습니다."
        for answer in (u10, "어느 사업인지 알려 주십시오.", "사업명을 알려주시면 다시 찾아보겠습니다."):
            with self.subTest(answer=answer):
                self.assertTrue(is_refusal(answer))
        r = evaluate([{"question": "그 사업 예산은?", "answerable": False}], lambda q, f: answer_with(u10))
        self.assertEqual(r["summary"]["refusal_accuracy"], 1.)
        # 정상 답변 끝에 되묻는 문장이 붙은 경우는 거절이 아님(첫 문장 기준 유지)
        self.assertFalse(is_refusal("사업 범위는 학사 시스템 고도화입니다 [1]. 다른 사업이라면 사업명을 알려주십시오."))

    def test_unanswerable_without_gold_allows_values_quoted_from_cited_source(self):
        sources = [source(1, "x", "※ 사업 착수일은 2019.11.01로 가정하여 산정"), source(2, "y", "사업기간: 6개월")]
        reply = lambda text: {"question": "q", "answer": text, "sources": sources, "model": "fake", "status": "ok"}
        case = {"question": "2019년에 공고된 사업은?", "answerable": False}
        for text, expected in [("2019년에 공고된 사업은 문서에서 확인되지 않습니다. 다만 출처에는 착수일을 2019.11.01로 가정했습니다 [1].", 1.),
                               ("2019년에 공고된 사업은 확인되지 않습니다. 다만 착수일은 2019.11.01입니다 [2].", 0.),  # 인용한 [2]에 없음
                               ("2019년에 공고된 사업은 확인되지 않습니다. 다만 착수일은 2020.03.01입니다 [1].", 0.)]:  # 지어낸 날짜
            with self.subTest(text=text):
                self.assertEqual(evaluate([case], lambda q, f: reply(text))["summary"]["refusal_accuracy"], expected)


class AggregationTests(unittest.TestCase):
    def test_dedup_ci_status_latency_and_breakdown(self):
        cases = [{"id": f"c{i}", "question": f"q{i}", "expected_doc_ids": ["a" if i < 3 else "b"], "grading": "keyword",
                  "expected_keywords": ["정답"], "weight": 1 / 3 if i < 3 else 1., "doc_cluster": "a" if i < 3 else "b",
                  "question_type": "금액" if i < 3 else "기간"} for i in range(4)]
        answers = iter([answer_with("정답 [1]", "a", texts=["정답 근거"]), answer_with("정답 [1]", "a"),
                        answer_with("정답 [1]", "a"), answer_with("오답", "x", status="truncated")])
        result = evaluate(cases, lambda q, f: next(answers))
        s = result["summary"]
        self.assertAlmostEqual(s["answer_accuracy"], .75)
        self.assertAlmostEqual(s["answer_accuracy_dedup"], .5)  # 같은 문서·측면 3문항은 1문항 몫
        self.assertEqual((s["status_ok_count"], s["status_truncated_count"], s["truncated_rate"]), (3, 1, .25))
        self.assertAlmostEqual(s["mrr_at_k"], .75)
        self.assertAlmostEqual(s["answer_chunk_hit_rate"], .25)
        self.assertIsNotNone(s["latency_p95_seconds"])
        self.assertEqual(result["breakdown"]["question_type"]["기간"]["accuracy"], 0.)
        self.assertEqual(result["breakdown"]["status"], {"ok": 3, "truncated": 1})
        answers = iter([answer_with("정답 [1]", "a", texts=["정답 근거"]), answer_with("정답 [1]", "a"),
                        answer_with("정답 [1]", "a"), answer_with("오답", "x", status="truncated")])
        again = evaluate(cases, lambda q, f: next(answers))["summary"]
        self.assertEqual(again["answer_accuracy_dedup_ci95_low"], s["answer_accuracy_dedup_ci95_low"])  # 시드 고정

    def test_case_token_usage_from_active_trace(self):
        trace = Trace({}, FakeLangfuse())

        def answer_fn(question, filters):
            model_call("generation", "gpt-5-mini", lambda: SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120)))
            return answer_with("답", "a")

        with trace.run("t"):
            result = evaluate([{"question": "q1", "expected_doc_ids": ["a"]}, {"question": "q2", "expected_doc_ids": ["a"]}],
                              answer_fn)
        self.assertEqual(result["records"][1]["case_token_usage"], {"gpt-5-mini": {"input": 100, "output": 20, "total": 120}})
        self.assertEqual((result["summary"]["input_tokens_per_case"], result["summary"]["output_tokens_per_case"]), (100, 20))


class JudgeTests(unittest.TestCase):
    def test_model_whitelist_cache_and_no_text_in_cache(self):
        with self.assertRaises(ValueError):
            make_llm_judge(None, model="gpt-4o")
        calls = []
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: calls.append(kw) or SimpleNamespace(
            output_text='{"verdict": "correct", "reason": "일치"}', usage=None)))
        case = {"id": "c1", "question": "PRIVATE_QUESTION", "reference_excerpt": "PRIVATE_TEXT", "evidence": [{"사업명": "사업"}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judge.jsonl"
            judge = make_llm_judge(client, cache_path=path)
            self.assertEqual(judge(case, "PRIVATE_ANSWER"), "correct")
            self.assertEqual(judge(case, "PRIVATE_ANSWER"), "correct")
            self.assertEqual(make_llm_judge(client, cache_path=path)(case, "PRIVATE_ANSWER"), "correct")
            self.assertEqual(len(calls), 1)
            self.assertNotIn("PRIVATE", path.read_text(encoding="utf-8"))
        self.assertEqual((calls[0]["model"], calls[0]["store"], calls[0]["reasoning"]), ("gpt-5-nano", False, {"effort": "low"}))

    def test_bad_judge_output_fails(self):
        client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text="모름", usage=None)))
        with self.assertRaises(ValueError):
            make_llm_judge(client)({"question": "q"}, "답")


if __name__ == "__main__":
    unittest.main()
