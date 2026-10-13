
import os
import chromadb

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag_lab.retrievers.base import BaseRetriever, RetrievalResult
from rag_lab.retrievers.rankings import (
    mmr_select,
    cosine_similarity
)


class MMRRetriever(BaseRetriever):

    def __init__(self, lambda_mult=0.5, fetch_k=20):
        load_dotenv()

        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")

        if fetch_k < 1:
            raise ValueError("fetch_k must be at least 1")

        self.lambda_mult = lambda_mult
        self.fetch_k = fetch_k

        # Connect to Chroma Cloud
        self.client = chromadb.CloudClient(
            api_key=os.environ["CHROMA_API_KEY"],
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"]
        )

        self.collection = self.client.get_collection(
            os.environ["CHROMA_COLLECTION"]
        )

        # Gemini embedding model
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=os.environ["GOOGLE_API_KEY"],
            output_dimensionality=768
        )

    def retrieve(self, query: str, k: int = 5):

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if k < 1:
            raise ValueError("k must be at least 1")

        # 1. Generate query embedding
        query_vector = self.embeddings.embed_query(query)

        # 2. Retrieve candidate chunks from Chroma
        response = self.collection.query(
            query_embeddings=[query_vector],
            n_results=max(k, self.fetch_k),
            include=[
                "documents",
                "metadatas",
                "embeddings"
            ]
        )

        ids = response["ids"][0]
        documents = response["documents"][0]
        metadata = response["metadatas"][0]
        vectors = response["embeddings"][0]

        if not ids:
            return []

        # 3. Prepare candidates for the tested MMR function
        candidates = [
            {
                "id": chunk_id,
                "embedding": vector
            }
            for chunk_id, vector in zip(ids, vectors)
        ]

        # 4. Select chunks using the shared ranking algorithm
        selected_ids = mmr_select(
            query_vector=query_vector,
            candidates=candidates,
            k=k,
            lambda_mult=self.lambda_mult
        )

        # 5. Map original Chroma IDs back to their data
        chunk_lookup = {}

        for i, chunk_id in enumerate(ids):
            chunk_lookup[chunk_id] = {
                "text": documents[i],
                "metadata": metadata[i] or {},
                "embedding": vectors[i]
            }

        # 6. Construct results in MMR ranking order
        results = []

        for chunk_id in selected_ids:
            chunk = chunk_lookup[chunk_id]

            relevance_score = cosine_similarity(
                query_vector,
                chunk["embedding"]
            )

            results.append(
                RetrievalResult(
                    id=chunk_id,
                    text=chunk["text"],
                    metadata=chunk["metadata"],
                    score=float(relevance_score)
                )
            )

        return results
