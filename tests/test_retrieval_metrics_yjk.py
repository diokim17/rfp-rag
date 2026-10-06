"""검색 평가 지표 Recall@k·MRR@k의 정의를 고정합니다 (외부 API·파일 없이)."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from retrieval_metrics_yjk import mean_scores, mrr_at_k, recall_at_k, rescore, score_case  # noqa: E402


class RecallAtKTest(unittest.TestCase):
    def test_counts_distinct_gold_documents_within_k(self):
        retrieved = ["a", "a", "x", "b", "y"]
        self.assertEqual(recall_at_k(["a", "b"], retrieved, 3), 0.5)  # b는 4위라 k=3 밖
        self.assertEqual(recall_at_k(["a", "b"], retrieved, 5), 1.0)
        self.assertEqual(recall_at_k(["a", "b", "c"], retrieved, 5), 2 / 3)

    def test_duplicate_chunks_of_one_document_count_once(self):
        self.assertEqual(recall_at_k(["a", "b"], ["a", "a", "a"], 3), 0.5)

    def test_short_or_empty_results(self):
        self.assertEqual(recall_at_k(["a"], ["a"], 10), 1.0)
        self.assertEqual(recall_at_k(["a"], [], 5), 0.0)


class MrrAtKTest(unittest.TestCase):
    def test_uses_first_gold_rank(self):
        self.assertEqual(mrr_at_k(["b"], ["x", "b", "b"], 5), 0.5)
        self.assertEqual(mrr_at_k(["a", "b"], ["x", "y", "b", "a"], 5), 1 / 3)

    def test_zero_when_gold_outside_k(self):
        self.assertEqual(mrr_at_k(["b"], ["x", "y", "b"], 2), 0.0)
        self.assertEqual(mrr_at_k(["b"], [], 5), 0.0)


class ValidationTest(unittest.TestCase):
    def test_rejects_empty_gold_and_bad_k(self):
        with self.assertRaises(ValueError):
            recall_at_k([], ["a"], 5)
        with self.assertRaises(ValueError):
            mrr_at_k(["a"], ["a"], 0)
        with self.assertRaises(ValueError):
            mean_scores([])


class AggregateTest(unittest.TestCase):
    def test_mean_scores_and_misses(self):
        rows = [{"id": "q1", **score_case(["a"], ["a"], 5)}, {"id": "q2", **score_case(["a", "b"], ["x", "a"], 5)}]
        summary = mean_scores(rows)
        self.assertEqual(summary["recall"], 0.75)
        self.assertEqual(summary["mrr"], 0.75)
        self.assertEqual(summary["misses"], ["q2"])

    def test_rescore_matches_saved_summary(self):
        cases = [{"id": "q1", "expected_doc_ids": ["a"]}, {"id": "q2", "expected_doc_ids": ["a", "b"]}]
        result = {"summary": [{"rerank": "none", "candidates": None, "top_k": 2, "recall": 0.75, "mrr": 0.75,
                               "mean_ms": 1.0, "median_ms": 1.0}],
                  "records": [{"rerank": "none", "candidates": None, "top_k": 2, "id": "q1", "hits": ["a", "x"]},
                              {"rerank": "none", "candidates": None, "top_k": 2, "id": "q2", "hits": ["x", "a"]}]}
        [row] = rescore(result, cases)
        self.assertTrue(row["matches_saved"])
        self.assertEqual(row["misses"], ["q2"])


if __name__ == "__main__":
    unittest.main()
