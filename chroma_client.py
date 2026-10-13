
import os
from functools import lru_cache

import chromadb
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_chroma_client():
    return chromadb.CloudClient(
        api_key=os.environ["CHROMA_API_KEY"],
        tenant=os.environ["CHROMA_TENANT"],
        database=os.environ["CHROMA_DATABASE"],
    )


@lru_cache(maxsize=1)
def get_chroma_collection():
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=os.getenv(
            "CHROMA_COLLECTION",
            "viper_knowledge_base"
        )
    )
