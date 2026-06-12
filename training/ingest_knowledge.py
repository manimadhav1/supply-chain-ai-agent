import os
import chromadb
from chromadb.utils import embedding_functions

def ingest_docs():
    # 1. Initialize local ChromaDB storage
    chroma_client = chromadb.PersistentClient(path="data/chroma_db")
    
    # 2. Use a free, local embedding model
    default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create or get a collection for your supply chain files
    collection = chroma_client.get_or_create_collection(
        name="supply_chain_kb", 
        embedding_function=default_ef
    )
    
    kb_dir = "knowledge_base"
    if not os.path.exists(kb_dir):
        print(f"Error: '{kb_dir}' folder not found. Run your generation script first!")
        return

    documents = []
    metadatas = []
    ids = []
    
    # 3. Read every generated text file
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple metadata extraction from filename (e.g., SOP_Coke500_Bangalore.txt)
            parts = filename.replace(".txt", "").split("_")
            sku = parts[1] if len(parts) > 1 else "Unknown"
            region = parts[2] if len(parts) > 2 else "Unknown"
            
            documents.append(content)
            metadatas.append({"sku": sku, "region": region, "source": filename})
            ids.append(filename)
            
    # 4. Upsert data into Vector DB
    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
        print(f"Successfully vectorized and indexed {len(documents)} documents into ChromaDB!")
    else:
        print("No text documents found to index.")

if __name__ == "__main__":
    ingest_docs()