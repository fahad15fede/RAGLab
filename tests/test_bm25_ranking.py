
import unittest
from rank_bm25 import BM25Okapi


class TestBM25Ranking(unittest.TestCase):

    def setUp(self):
        self.documents = [
            "fang age integer declaration syntax",
            "strike shift slither conditional statement",
            "illegal identifier cannot start with digit",
            "venom price floating point variable",
            "fang marks integer declaration example",
            "coil name string variable declaration",
        ]

        self.tokenized = [
            document.lower().split()
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        return sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i)
        )

    def test_integer_declaration(self):
        ranked = self.search("fang integer")

        self.assertIn(ranked[0], [0, 4])

    def test_conditional_keywords(self):
        ranked = self.search("strike shift slither")

        self.assertEqual(ranked[0], 1)

    def test_illegal_identifier(self):
        ranked = self.search("illegal identifier")

        self.assertEqual(ranked[0], 2)

    def test_float_datatype(self):
        ranked = self.search("venom floating")

        self.assertEqual(ranked[0], 3)


if __name__ == "__main__":
    unittest.main()
