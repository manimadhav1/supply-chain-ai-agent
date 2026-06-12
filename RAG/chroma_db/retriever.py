from functools import lru_cache
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

client = PersistentClient(
    path="RAG/chroma_db_v2"
)

collection_name = "enterprise_memory_v2"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

existing_collections = [col.name for col in client.list_collections()]
if collection_name in existing_collections:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )

@lru_cache(maxsize=100)
def retrieve(query, n_results=10):
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