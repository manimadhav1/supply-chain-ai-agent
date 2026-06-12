import pandas as pd

inventory_df = pd.read_csv(
    "data/inventory.csv"
)

def inventory_tool(sku):

    result = inventory_df[
        inventory_df["sku"] == sku
    ]

    if result.empty:
        raise ValueError(
            f"{sku} not found"
        )

    stock = int(
        result.iloc[0]["stock"]
    )

    return stock