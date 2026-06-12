import pandas as pd

# Load datasets
sales = pd.read_csv("data/sales.csv")
inventory = pd.read_csv("data/inventory.csv")
notes = pd.read_csv("data/sales_notes.csv")

memory = []

record_id = 1

# ========================================
# INVENTORY RECORDS
# ========================================

for _, row in inventory.iterrows():

    memory.append({
        "record_id": record_id,
        "record_type": "inventory",
        "sku": row["sku"],
        "document": f"Current inventory for {row['sku']} is {row['stock']} units."
    })

    record_id += 1

# ========================================
# SALES NOTES
# ========================================

for _, row in notes.iterrows():

    memory.append({
        "record_id": record_id,
        "record_type": "sales_note",
        "sku": row["sku"],
        "document": row["note"]
    })

    record_id += 1

# ========================================
# AVERAGE SALES PER SKU
# ========================================

sku_avg = sales.groupby("sku")["sales"].mean()

for sku, avg_sales in sku_avg.items():

    memory.append({
        "record_id": record_id,
        "record_type": "avg_sales",
        "sku": sku,
        "document": f"Average daily sales for {sku} are {round(avg_sales)} units."
    })

    record_id += 1

# ========================================
# BEST REGION PER SKU
# ========================================

region_sales = (
    sales.groupby(["sku", "region"])["sales"]
    .mean()
    .reset_index()
)

for sku in sales["sku"].unique():

    temp = region_sales[region_sales["sku"] == sku]

    best = temp.loc[temp["sales"].idxmax()]

    memory.append({
        "record_id": record_id,
        "record_type": "best_region",
        "sku": sku,
        "document":
        f"{best['region']} is the best performing region for {sku} with average sales of {round(best['sales'])} units."
    })

    record_id += 1

# ========================================
# PEAK SALES
# ========================================

for sku in sales["sku"].unique():

    sku_data = sales[sales["sku"] == sku]

    peak = sku_data.loc[sku_data["sales"].idxmax()]

    memory.append({
        "record_id": record_id,
        "record_type": "peak_sales",
        "sku": sku,
        "document":
        f"Highest sales for {sku} were {peak['sales']} units on {peak['date']}."
    })

    record_id += 1

# ========================================
# PROMOTION IMPACT
# ========================================

for sku in sales["sku"].unique():

    sku_data = sales[sales["sku"] == sku]

    promo_avg = sku_data[sku_data["promo"] == 1]["sales"].mean()
    normal_avg = sku_data[sku_data["promo"] == 0]["sales"].mean()

    uplift = ((promo_avg - normal_avg) / normal_avg) * 100

    memory.append({
        "record_id": record_id,
        "record_type": "promotion_analysis",
        "sku": sku,
        "document":
        f"Promotions increase {sku} sales by approximately {round(uplift,1)} percent."
    })

    record_id += 1

# ========================================
# EVENT IMPACT
# ========================================

for sku in sales["sku"].unique():

    sku_data = sales[sales["sku"] == sku]

    event_avg = sku_data[sku_data["event"] == 1]["sales"].mean()
    normal_avg = sku_data[sku_data["event"] == 0]["sales"].mean()

    uplift = ((event_avg - normal_avg) / normal_avg) * 100

    memory.append({
        "record_id": record_id,
        "record_type": "event_analysis",
        "sku": sku,
        "document":
        f"Events increase {sku} sales by approximately {round(uplift,1)} percent."
    })

    record_id += 1

# ========================================
# TEMPERATURE ANALYSIS
# ========================================

for sku in sales["sku"].unique():

    sku_data = sales[sales["sku"] == sku]

    hot = sku_data[sku_data["temp"] >= 35]["sales"].mean()
    normal = sku_data[sku_data["temp"] < 35]["sales"].mean()

    uplift = ((hot - normal) / normal) * 100

    memory.append({
        "record_id": record_id,
        "record_type": "temperature_analysis",
        "sku": sku,
        "document":
        f"When temperatures exceed 35°C, {sku} sales increase by approximately {round(uplift,1)} percent."
    })

    record_id += 1
    # ========================================
# HIGHEST SELLING SKU
# ========================================

avg_sales = sales.groupby("sku")["sales"].mean()

best_sku = avg_sales.idxmax()
best_sales = avg_sales.max()

memory.append({
    "record_id": record_id,
    "record_type": "top_sku",
    "sku": best_sku,
    "document":
    f"{best_sku} is the highest selling SKU with average daily sales of {round(best_sales)} units."
})

record_id += 1

# ========================================
# LOWEST SELLING SKU
# ========================================

worst_sku = avg_sales.idxmin()
worst_sales = avg_sales.min()

memory.append({
    "record_id": record_id,
    "record_type": "lowest_sku",
    "sku": worst_sku,
    "document":
    f"{worst_sku} is the lowest selling SKU with average daily sales of {round(worst_sales)} units."
})

record_id += 1

# ========================================
# INVENTORY RANKINGS
# ========================================

highest_stock = inventory.loc[
    inventory["stock"].idxmax()
]

memory.append({
    "record_id": record_id,
    "record_type": "inventory_ranking",
    "sku": highest_stock["sku"],
    "document":
    f"{highest_stock['sku']} currently has the highest inventory with {highest_stock['stock']} units."
})

record_id += 1


lowest_stock = inventory.loc[
    inventory["stock"].idxmin()
]

memory.append({
    "record_id": record_id,
    "record_type": "inventory_ranking",
    "sku": lowest_stock["sku"],
    "document":
    f"{lowest_stock['sku']} currently has the lowest inventory with {lowest_stock['stock']} units."
})

record_id += 1

# ========================================
# MONTHLY SALES TRENDS
# ========================================

sales["date"] = pd.to_datetime(sales["date"])

sales["month"] = sales["date"].dt.month

monthly = (
    sales.groupby(["sku", "month"])["sales"]
    .mean()
    .reset_index()
)

for _, row in monthly.iterrows():

    memory.append({
        "record_id": record_id,
        "record_type": "monthly_trend",
        "sku": row["sku"],
        "document":
        f"Average sales for {row['sku']} in month {int(row['month'])} were {round(row['sales'])} units."
    })

    record_id += 1

    # ========================================
# PRODUCT COMPARISON
# ========================================

avg_sales = sales.groupby("sku")["sales"].mean()

sku_list = list(avg_sales.index)

for i in range(len(sku_list)):

    for j in range(i + 1, len(sku_list)):

        sku1 = sku_list[i]
        sku2 = sku_list[j]

        diff = (
            (avg_sales[sku1] - avg_sales[sku2])
            / avg_sales[sku2]
        ) * 100

        memory.append({
            "record_id": record_id,
            "record_type": "comparison",
            "sku": sku1,
            "document":
            f"{sku1} sells {abs(round(diff,1))}% {'more' if diff > 0 else 'less'} than {sku2}."
        })

        record_id += 1

# ========================================
# SAVE MEMORY
# ========================================

memory_df = pd.DataFrame(memory)

memory_df.to_csv(
    "data/enterprise_memory.csv",
    index=False
)

print(f"Created {len(memory_df)} memory records")
print(memory_df.head())