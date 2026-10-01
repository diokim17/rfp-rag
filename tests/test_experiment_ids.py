from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest

from experiment_ids import next_experiment_id, owner_initials


class ExperimentIdTests(unittest.TestCase):
    def test_persistent_counters_and_owner_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(next_experiment_id("DYK", directory), "dyk-0001")
            self.assertEqual(next_experiment_id("김도영", directory), "dyk-0002")
            self.assertEqual(next_experiment_id("yjk", directory), "yjk-0001")
            self.assertEqual(next_experiment_id("dyk", directory), "dyk-0003")

    def test_concurrent_allocations_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            with ThreadPoolExecutor(max_workers=8) as pool:
                ids = list(pool.map(lambda _: next_experiment_id("dyk", directory), range(20)))
            self.assertEqual(set(ids), {f"dyk-{i:04d}" for i in range(1, 21)})

    def test_invalid_owner_does_not_become_shared_default(self):
        for owner in ("", "../dyk", "김아무개", "spai1313"):
            with self.assertRaises(ValueError):
                owner_initials(owner)


if __name__ == "__main__":
    unittest.main()
