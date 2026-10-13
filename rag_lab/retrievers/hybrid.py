
from rag_lab.retrievers.base import BaseRetriever, RetrievalResult
from rag_lab.retrievers.semantic import SemanticRetriever
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.rankings import reciprocal_rank_fusion


class HybridRetriever(BaseRetriever):

    def __init__(self, rrf_k=60):
        if rrf_k < 0:
            raise ValueError("rrf_k must be non-negative")

        self.semantic = SemanticRetriever()
        self.bm25 = BM25Retriever()
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 5):

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k < 1:
            raise ValueError("k must be at least 1")

        # 1. Retrieve candidates from both methods
        candidate_k = max(k * 4, 20)

        semantic_results = self.semantic.retrieve(
            query,
            k=candidate_k
        )

        bm25_results = self.bm25.retrieve(
            query,
            k=candidate_k
        )

        # 2. Fuse rankings using our tested function
        ranked_ids = reciprocal_rank_fusion(
            [
                [result.id for result in bm25_results],
                [result.id for result in semantic_results]
            ],
            k=self.rrf_k
        )

        # 3. Map IDs back to original chunk content
        result_lookup = {
            result.id: result
            for result in semantic_results + bm25_results
        }

        # 4. Calculate RRF scores for returned chunks
        scores = {}

        for result_list in [bm25_results, semantic_results]:
            for rank, result in enumerate(result_list, start=1):
                scores[result.id] = (
                    scores.get(result.id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

        # 5. Return ranked results
        return [
            RetrievalResult(
                id=chunk_id,
                text=result_lookup[chunk_id].text,
                metadata=result_lookup[chunk_id].metadata,
                score=float(scores[chunk_id])
            )
            for chunk_id in ranked_ids[:k]
        ]
