
import unittest

from rag_lab.experiments.benchmark_v2 import (
    jaccard,
    percentile,
)


class TestRAGMetrics(unittest.TestCase):

    def test_jaccard_identical(self):
        a = {"chunk1", "chunk2"}
        b = {"chunk1", "chunk2"}

        self.assertEqual(jaccard(a, b), 1.0)

    def test_jaccard_no_overlap(self):
        a = {"chunk1", "chunk2"}
        b = {"chunk3", "chunk4"}

        self.assertEqual(jaccard(a, b), 0.0)

    def test_jaccard_partial_overlap(self):
        a = {"chunk1", "chunk2", "chunk3"}
        b = {"chunk2", "chunk3", "chunk4"}

        self.assertAlmostEqual(jaccard(a, b), 0.5)

    def test_percentile(self):
        values = [10, 20, 30, 40, 50]

        self.assertEqual(percentile(values, 50), 30)
        self.assertEqual(percentile(values, 0), 10)
        self.assertEqual(percentile(values, 100), 50)

    def test_percentile_single_value(self):
        self.assertEqual(percentile([25], 95), 25)


if __name__ == "__main__":
    unittest.main()
