import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)
#above ^ are changes made by {neeraj_k} for rag
from workflow.graph import graph

state = {

    "sku":"Coke500",

    "region":"Bangalore",

    "temp":39,

    "promo":1,

    "event":1,

    "month":6,

    "dayofweek":5,

    "sales_note":
    """
    IPL finals expected to
    increase beverage sales.

    Retailers expecting
    strong demand.
    """
}

result = graph.invoke(state)

print("\n")
print("=" * 60)

print("SUPPLY CHAIN ANALYSIS REPORT")

print("=" * 60)

print(f"SKU: {result['sku']}")
print(f"Region: {result['region']}")

print("-" * 60)

print(f"Demand Boost: {result['demand_boost']}%")
print(f"Forecast: {round(result['forecast'])} units")
print(f"Inventory: {result['inventory']} units")

print("-" * 60)

print(f"Risk Level: {result['risk']}")
print(f"Days Cover: {result['days_cover']}")
print(f"Shortage: {round(result['shortage'])}")

print("-" * 60)

print(f"Recommended Reorder: {result['reorder_qty']} units")

print("-" * 60)

print("AI SUMMARY:")

print(result["summary"])

print("=" * 60)