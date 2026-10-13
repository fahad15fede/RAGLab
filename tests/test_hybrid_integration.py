
import unittest
from unittest.mock import patch

from rag_lab.retrievers.base import RetrievalResult
from rag_lab.retrievers.hybrid import HybridRetriever


def make_result(chunk_id):
    return RetrievalResult(
        id=chunk_id,
        text=f"Content for {chunk_id}",
        metadata={"filename": "test.pdf"},
        score=1.0
    )


class TestHybridIntegration(unittest.TestCase):

    @patch("rag_lab.retrievers.hybrid.BM25Retriever")
    @patch("rag_lab.retrievers.hybrid.SemanticRetriever")
    def test_hybrid_ranking(self, mock_semantic, mock_bm25):

        mock_bm25.return_value.retrieve.return_value = [
            make_result("A"),
            make_result("B"),
            make_result("C")
        ]

        mock_semantic.return_value.retrieve.return_value = [
            make_result("B"),
            make_result("D"),
            make_result("A")
        ]

        retriever = HybridRetriever()
        results = retriever.retrieve(
            "How do I declare a fang variable?",
            k=3
        )

        self.assertEqual(
            [result.id for result in results],
            ["B", "A", "D"]
        )

        self.assertGreater(
            results[0].score,
            results[1].score
        )

        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
