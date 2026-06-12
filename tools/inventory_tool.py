import pandas as pd

inventory_df = pd.read_excel(
    "data/FMCG_DemandForecasting_Dataset.xlsx",
    sheet_name="Inventory_Snapshot",
    header=1
)

inventory_df = inventory_df.rename(columns={
    "SKU ID": "sku",
    "Region Warehouse": "region_warehouse",
    "Current Stock (Units)": "current_stock"
})

inventory_df["sku"] = inventory_df["sku"].astype(str).str.strip()
inventory_df["region_warehouse"] = inventory_df["region_warehouse"].astype(str).str.strip()


def inventory_tool(sku, region=None):
    sku = str(sku).strip()
    region = str(region).strip() if region is not None else ""

    matches = inventory_df[inventory_df["sku"] == sku]
    if matches.empty:
        raise ValueError(f"SKU {sku} not found in inventory snapshot")

    if region:
        region_matches = matches[matches["region_warehouse"].str.lower().str.startswith(region.lower())]
        if not region_matches.empty:
            stock = int(region_matches.iloc[0]["current_stock"])
            return stock

    # Fallback to the first available warehouse for the SKU
    stock = int(matches.iloc[0]["current_stock"])
    return stock