import pandas as pd

_df = pd.read_csv("data/inventory.csv")

def inventory_tool(sku: str) -> int:
    result = _df[_df["sku"] == sku]
    if result.empty:
        raise ValueError(f"SKU '{sku}' not found in inventory")
    return int(result.iloc[0]["stock"])
