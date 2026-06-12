import pandas as pd
import time

EXCEL_PATH = "data/FMCG_DemandForecasting_Dataset.xlsx"
DB_PATH = "RAG/chroma_db_v2"
COLLECTION_NAME = "enterprise_memory_v2"

sheets = [
    "Product_Master",
    "Inventory_Snapshot",
    "Monthly_KPI_Summary",
    "Demand_Forecast",
    "Supplier_Master",
    "User_Stories_RAG",
    "Sales_Transactions",
    "README_DataDict"
]


def get_client_and_collection():
    from chromadb import PersistentClient
    from chromadb.utils import embedding_functions

    start = time.time()
    client = PersistentClient(path=DB_PATH)
    print(f"ChromaDB client initialized in {time.time() - start:.2f}s")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    start = time.time()
    existing_collections = [col.name for col in client.list_collections()]
    print(f"Listed {len(existing_collections)} existing collections in {time.time() - start:.2f}s")

    if COLLECTION_NAME in existing_collections:
        collection = client.get_collection(name=COLLECTION_NAME)
    else:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn
        )

    print(f"Collection ready: {COLLECTION_NAME}")
    return collection


def add_document_from_row(collection, sheet, idx, row_data):
    document_text = "\n".join(
        [f"{key}: {value}" for key, value in row_data.items()]
    )
    metadata = {
        "source": sheet,
        "sheet": sheet,
        "row_index": idx,
        **{k: str(v) for k, v in row_data.items() if k in ["SKU ID", "Region", "Top SKU", "Top Region", "Query Type", "Channel", "Category"]}
    }
    collection.add(
        ids=[f"{sheet}_{idx}"],
        documents=[document_text],
        metadatas=[metadata]
    )


def add_summary_document(collection, summary_id, title, text, metadata=None):
    collection.add(
        ids=[summary_id],
        documents=[f"{title}\n\n{text}"],
        metadatas=[metadata or {"source": "summary"}]
    )


def build_collection():
    print("Starting RAG loader... this may take a moment while libraries initialize.")
    print("Loading data and ChromaDB dependencies...")

    collection = get_client_and_collection()
    memory = pd.read_csv("data/enterprise_memory.csv")

    # Clear old records if rerunning
    try:
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
    except Exception:
        pass

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

    doc_count = 0
    for sheet in sheets:
        print(f"Reading sheet {sheet}...")
        try:
            df = pd.read_excel(EXCEL_PATH, sheet_name=sheet, header=1)
            print(f"  loaded {len(df)} rows from {sheet}")
        except Exception as e:
            print(f"  failed to read {sheet}: {e}")
            continue

        # Add each row as a document
        for idx, row in df.iterrows():
            row_data = row.dropna().to_dict()
            if not row_data:
                continue
            add_document_from_row(collection, sheet, idx, row_data)
            doc_count += 1

        # Add helpful, high-level summary documents for Sales_Transactions
        if sheet == "Sales_Transactions":
            try:
                sales_df = df.copy()
                sales_df = sales_df.dropna(subset=["Region", "Channel", "Qty Sold"])
                region_totals = (
                    sales_df.groupby("Region")["Qty Sold"]
                    .sum()
                    .sort_values(ascending=False)
                )
                region_text = "\n".join(
                    [f"{region}: {int(total)} units sold" for region, total in region_totals.items()]
                )
                add_summary_document(
                    collection,
                    "summary_region_demand",
                    "Region Demand Summary",
                    region_text,
                    {"source": "Sales_Transactions", "summary_type": "region_demand"}
                )

                channel_totals = (
                    sales_df.groupby("Channel")["Qty Sold"]
                    .sum()
                    .sort_values(ascending=False)
                )
                channel_text = "\n".join(
                    [f"{channel}: {int(total)} units sold" for channel, total in channel_totals.items()]
                )
                add_summary_document(
                    collection,
                    "summary_channel_demand",
                    "Channel Demand Summary",
                    channel_text,
                    {"source": "Sales_Transactions", "summary_type": "channel_demand"}
                )

                highest_region = region_totals.index[0]
                highest_region_text = f"Highest demand region: {highest_region} with {int(region_totals.iloc[0])} units sold."
                add_summary_document(
                    collection,
                    "summary_highest_region",
                    "Highest Demand Region",
                    highest_region_text,
                    {"source": "Sales_Transactions", "summary_type": "highest_region"}
                )

                top_skus = (
                    sales_df.groupby("SKU ID")["Qty Sold"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                )
                sku_text = "\n".join(
                    [f"{sku}: {int(total)} units sold" for sku, total in top_skus.items()]
                )
                add_summary_document(
                    collection,
                    "summary_top_skus",
                    "Top SKU Demand Summary",
                    sku_text,
                    {"source": "Sales_Transactions", "summary_type": "top_skus"}
                )
            except Exception:
                pass

    print(f"Loaded {len(memory)} enterprise records and {doc_count} dataset rows into ChromaDB")
    print("Added Sales_Transactions summaries for region, channel, and top SKUs.")


if __name__ == "__main__":
    build_collection()
