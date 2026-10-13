import unittest

from evaluation import _normalize, evaluate


def answers(mapping):
    return lambda question, filters: {"answer": mapping[question], "sources": [{"doc_id": "a"}]}


class NormalizeTest(unittest.TestCase):
    def test_won_units_and_commas(self):
        for text, number in (("1억 원", "100000000"), ("100,000,000원", "100000000"), ("8,700만 원", "87000000"),
                             ("2억 4,290만 원", "242900000"), ("2억4천만", "240000000"), ("3.52억", "352000000")):
            with self.subTest(text=text):
                self.assertIn(number, _normalize(text))

    def test_dates(self):
        for text in ("2025년 5월 31일", "2025. 5. 31.", "2025.05.31", "2025-05-31"):
            with self.subTest(text=text):
                self.assertIn("2025-05-31", _normalize(text))

    def test_other_keywords_unchanged(self):
        for word in ("20분", "1.25", "210일", "80점"):
            self.assertEqual(_normalize(word), word)


class EvaluateTest(unittest.TestCase):
    cases = [{"question": "예산", "expected_doc_ids": ["a"], "expected_keywords": ["100,000,000"]},
             {"question": "기한", "expected_doc_ids": ["a"], "expected_keywords": ["2025-05-31"]},
             {"question": "없는 사업", "expected_doc_ids": [], "answerable": False},
             {"question": "낙찰 업체", "expected_doc_ids": ["a"], "answerable": False}]

    def test_no_answer_cases_scored_by_refusal_only(self):
        result = evaluate(self.cases, answers({"예산": "사업 예산은 1억 원입니다.", "기한": "2025년 5월 31일까지입니다.",
                                               "없는 사업": "질문과 관련된 RFP 문서가 없어 답변할 수 없습니다.",
                                               "낙찰 업체": "낙찰 업체는 ○○입니다."}))
        self.assertEqual(result["summary"]["recall_at_k"], 1)
        self.assertEqual(result["summary"]["keyword_coverage"], 1)
        self.assertEqual((result["summary"]["answerable_count"], result["summary"]["no_answer_count"]), (2, 2))
        self.assertEqual(result["summary"]["refusal_accuracy"], 0.5)
        self.assertEqual([r["refused"] for r in result["records"]], [None, None, True, False])

    def test_summary_unchanged_without_no_answer_cases(self):
        result = evaluate(self.cases[:1], answers({"예산": "1억 원"}))
        self.assertEqual(set(result["summary"]), {"count", "recall_at_k", "keyword_coverage", "latency_seconds"})

    def test_empty_doc_ids_still_rejected_for_answerable_case(self):
        with self.assertRaises(ValueError):
            evaluate([{"question": "예산", "expected_doc_ids": []}], None)


if __name__ == "__main__":
    unittest.main()
