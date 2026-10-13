
import os
import chromadb

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from rag_lab.retrievers.base import (
    BaseRetriever,
    RetrievalResult
)


class SemanticRetriever(BaseRetriever):

    def __init__(self):
        load_dotenv()

        # Connect to the existing Chroma Cloud database
        client = chromadb.CloudClient(
            api_key=os.environ["CHROMA_API_KEY"],
            tenant=os.environ["CHROMA_TENANT"],
            database=os.environ["CHROMA_DATABASE"]
        )

        self.collection = client.get_collection(
            os.environ["CHROMA_COLLECTION"]
        )

        # Same embedding configuration as original VIPERBot
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

        # Step 1: Embed the user's question
        query_vector = self.embeddings.embed_query(query)

        # Step 2: Find nearest vectors in Chroma
        response = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        # Step 3: Convert to our common result format
        results = []

        for i, chunk_id in enumerate(response["ids"][0]):
            results.append(
                RetrievalResult(
                    id=chunk_id,
                    text=response["documents"][0][i],
                    metadata=response["metadatas"][0][i] or {},
                    score=float(response["distances"][0][i])
                )
            )

        return results
