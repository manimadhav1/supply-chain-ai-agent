import pandas as pd
from chromadb import PersistentClient

# Load memory
memory = pd.read_csv(
    "data/enterprise_memory.csv"
)

# Create database
client = PersistentClient(
    path="rag/chroma_db"
)

# Create collection
collection = client.get_or_create_collection(
    name="enterprise_memory"
)

# Clear old records if rerunning
try:
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )
except:
    pass

# Add records
for _, row in memory.iterrows():

    collection.add(
        ids=[str(row["record_id"])],
        documents=[row["document"]],
        metadatas=[{
            "record_type":
            row["record_type"],

            "sku":
            row["sku"]
        }]
    )

print(
    f"Loaded {len(memory)} records into ChromaDB"
)