import chromadb
from chromadb.utils import embedding_functions

def query_knowledge_base(query_text, sku=None, region=None):
    """
    Queries the local vector DB for context. Optional metadata filtering by SKU/Region.
    """

    chroma_client = chromadb.PersistentClient(path="data/chroma_db")

    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = chroma_client.get_collection(
        name="supply_chain_kb",
        embedding_function=default_ef
    )

    # ----------------------------
    # FIX: Proper ChromaDB filter
    # ----------------------------
    filters = []

    if sku:
        filters.append({"sku": {"$eq": sku}})

    if region:
        filters.append({"region": {"$eq": region}})

    if len(filters) == 0:
        where_filter = None
    elif len(filters) == 1:
        where_filter = filters[0]
    else:
        where_filter = {"$and": filters}

    # ----------------------------
    # Query vector DB
    # ----------------------------
    results = collection.query(
        query_texts=[query_text],
        n_results=2,
        where=where_filter
    )

    # ----------------------------
    # Flatten results safely
    # ----------------------------
    context = "\n\n".join(results.get("documents", [[]])[0])

    return context if context.strip() else "No specific organizational clauses found for this target."