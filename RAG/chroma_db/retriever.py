from functools import lru_cache
from chromadb import PersistentClient

client = PersistentClient(
    path="RAG/chroma_db"
)

collection = client.get_collection(
    name="enterprise_memory"
)

@lru_cache(maxsize=100)
def retrieve(query, n_results=3):

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    return results["documents"][0]