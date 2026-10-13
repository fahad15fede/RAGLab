
import unittest

from rag_lab.retrievers.rankings import (
    reciprocal_rank_fusion,
    cosine_similarity,
    mmr_select,
)


class TestRanking(unittest.TestCase):

    def test_rrf_favors_shared_chunks(self):
        bm25 = ["A", "B", "C"]
        semantic = ["B", "D", "A"]

        ranked = reciprocal_rank_fusion(
            [bm25, semantic]
        )

        self.assertEqual(ranked[0], "B")
        self.assertEqual(ranked[1], "A")

    def test_cosine_identical_vectors(self):
        self.assertAlmostEqual(
            cosine_similarity([1, 2], [1, 2]),
            1.0
        )

    def test_cosine_orthogonal_vectors(self):
        self.assertAlmostEqual(
            cosine_similarity([1, 0], [0, 1]),
            0.0
        )

    def test_mmr_relevance_only(self):
        query = [1, 0]

        candidates = [
            {"id": "A", "embedding": [1, 0]},
            {"id": "B", "embedding": [0.8, 0.6]},
            {"id": "C", "embedding": [0, 1]},
        ]

        selected = mmr_select(
            query,
            candidates,
            k=2,
            lambda_mult=1.0
        )

        self.assertEqual(selected, ["A", "B"])

    def test_mmr_diversity(self):
        query = [1, 0]

        candidates = [
            {"id": "A", "embedding": [1, 0]},
            {"id": "B", "embedding": [0.99, 0.1]},
            {"id": "C", "embedding": [0, 1]},
        ]

        selected = mmr_select(
            query,
            candidates,
            k=2,
            lambda_mult=0.3
        )

        self.assertEqual(selected, ["A", "C"])

    def test_invalid_lambda(self):
        with self.assertRaises(ValueError):
            mmr_select(
                [1, 0],
                [],
                lambda_mult=1.5
            )


if __name__ == "__main__":
    unittest.main()
