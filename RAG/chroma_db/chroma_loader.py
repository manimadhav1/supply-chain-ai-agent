import pandas as pd
from chromadb import PersistentClient
from pathlib import Path

# Prefer an Excel file if provided, otherwise fallback to CSV
data_xlsx = Path("data/enterprise_memory.xlsx")
data_csv = Path("data/enterprise_memory.csv")

if data_xlsx.exists():
    memory = pd.read_excel(data_xlsx)
elif data_csv.exists():
    memory = pd.read_csv(data_csv)
else:
    raise FileNotFoundError(
        "No enterprise memory file found. Place data/enterprise_memory.xlsx or data/enterprise_memory.csv"
    )

# Create database (ensure path matches repo folder name)
client = PersistentClient(
    path="RAG/chroma_db"
)

# Create collection
collection = client.get_or_create_collection(
    name="enterprise_memory"
)

# Clear old records if rerunning
try:
    existing = collection.get()

    if existing.get("ids"):
        collection.delete(
            ids=existing["ids"]
        )
except Exception:
    pass

# Add records — support multiple possible column names
for _, row in memory.iterrows():

    record_id = row.get("record_id") or row.get("id") or row.name

    document = row.get("document") or row.get("text") or ""

    metadata = {}
    if "record_type" in row:
        metadata["record_type"] = row["record_type"]
    if "sku" in row:
        metadata["sku"] = row["sku"]

    collection.add(
        ids=[str(record_id)],
        documents=[str(document)],
        metadatas=[metadata]
    )

print(f"Loaded {len(memory)} records into ChromaDB")