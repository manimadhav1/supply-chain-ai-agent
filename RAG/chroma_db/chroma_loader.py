import pandas as pd
from chromadb import PersistentClient

# Load enterprise memory
memory = pd.read_csv("data/enterprise_memory.csv")

# Create database
client = PersistentClient(
    path="RAG/chroma_db"
)

# Create or load collection
existing_collections = [col.name for col in client.list_collections()]
if "enterprise_memory" in existing_collections:
    collection = client.get_collection(name="enterprise_memory")
else:
    collection = client.create_collection(name="enterprise_memory")

# Clear old records if rerunning
try:
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
except Exception:
    pass

# Add enterprise memory records
for _, row in memory.iterrows():
    collection.add(
        ids=[f"enterprise_{row['record_id']}"],
        documents=[str(row["document"])],
        metadatas=[{
            "source": "enterprise_memory",
            "record_type": str(row.get("record_type", "")),
            "sku": str(row.get("sku", ""))
        }]
    )

# Ingest Excel dataset sheets for richer RAG data
excel_path = "data/FMCG_DemandForecasting_Dataset.xlsx"
workbook = pd.ExcelFile(excel_path)

sheets = [
    "Product_Master",
    "Inventory_Snapshot",
    "Monthly_KPI_Summary",
    "Demand_Forecast",
    "Supplier_Master",
    "User_Stories_RAG"
]

doc_count = 0
for sheet in sheets:
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet, header=1)
    except Exception:
        continue

    for idx, row in df.iterrows():
        row_data = row.dropna().to_dict()
        if not row_data:
            continue

        document_text = "\n".join(
            [f"{key}: {value}" for key, value in row_data.items()]
        )

        collection.add(
            ids=[f"{sheet}_{idx}"],
            documents=[document_text],
            metadatas=[{
                "source": sheet,
                "sheet": sheet,
                "row_index": idx,
                **{k: str(v) for k, v in row_data.items() if k in ["SKU ID", "Region", "Top SKU", "Top Region", "Query Type"]}
            }]
        )
        doc_count += 1

print(f"Loaded {len(memory)} enterprise records and {doc_count} dataset rows into ChromaDB")