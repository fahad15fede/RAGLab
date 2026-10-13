
import os
import re
import chromadb

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from rag_lab.retrievers.base import (
    BaseRetriever,
    RetrievalResult
)


class BM25Retriever(BaseRetriever):

    def __init__(self):
        load_dotenv()

        # Connect to existing Chroma Cloud
        self.client = chromadb.CloudClient(
            api_key=os.environ["CHROMA_API_KEY"],
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"]
        )

        self.collection = self.client.get_collection(
            os.environ["CHROMA_COLLECTION"]
        )

        # Load all stored chunks and metadata
        stored = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = stored["documents"]
        self.metadata = stored["metadatas"]
        self.ids = stored["ids"]

        if not self.documents:
            raise ValueError("Chroma collection is empty")

        # Tokenize all stored chunks
        tokenized_corpus = [
            self.tokenize(doc)
            for doc in self.documents
        ]

        # Build BM25 index once
        self.bm25 = BM25Okapi(tokenized_corpus)

        print(
            f"BM25 index ready: "
            f"{len(self.documents)} chunks"
        )

    @staticmethod
    def tokenize(text):
        return re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_]*|"
            r"\d+(?:\.\d+)?|"
            r"&&|\|\||==|!=|<=|>=|"
            r"[^\s]",
            text.lower()
        )

    def retrieve(self, query: str, k: int = 5):

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k < 1:
            raise ValueError("k must be at least 1")

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        # Sort by BM25 score, highest first
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], self.ids[i])
        )

        results = []

        for i in ranked_indices:

            # Ignore nonmatching chunks
            if scores[i] <= 0:
                continue

            results.append(
                RetrievalResult(
                    id=self.ids[i],
                    text=self.documents[i],
                    metadata=self.metadata[i] or {},
                    score=float(scores[i])
                )
            )

            if len(results) >= k:
                break

        return results
