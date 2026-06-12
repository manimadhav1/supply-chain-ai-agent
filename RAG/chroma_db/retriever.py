from functools import lru_cache
from chromadb import PersistentClient

client = PersistentClient(
    path="RAG/chroma_db"
)

collection = client.get_collection(
    name="enterprise_memory"
)

@lru_cache(maxsize=100)
def retrieve(query, n_results=5):
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas"]
    )

    docs = []
    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        docs.append({
            "document": document,
            "metadata": metadata
        })
    return docs