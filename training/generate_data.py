import pandas as pd
import numpy as np

np.random.seed(42)

skus = [
    "Coke500",
    "Sprite500",
    "Fanta500",
    "ThumsUp500",
]

regions = [
    "Bangalore",
    "Chennai",
    "Mumbai",
    "Hyderabad"
]

dates = pd.date_range(
    start="2025-01-01",
    end="2025-12-31"
)

rows = []

for sku in skus:

    for region in regions:

        base = np.random.randint(
            3000,
            7000
        )

        for date in dates:

            temp = np.random.randint(
                20,
                42
            )

            promo = np.random.choice(
                [0,1],
                p=[0.9,0.1]
            )

            event = np.random.choice(
                [0,1],
                p=[0.95,0.05]
            )

            weekend = (
                date.weekday() >= 5
            )

            day_of_year = date.timetuple().tm_yday

            seasonality = (
                1 +
                0.25*np.sin(
                    2*np.pi*day_of_year/365
                )
            )

            sales = base * seasonality

            sales += temp * 150
            sales += promo * 3000
            sales += event * 4500

            if date.month in [4,5,6]:
                sales*=1.25

            if weekend:
                sale*=1.15

            sales += np.random.normal(
                0,
                sales*0.05
            )
            
            sales += np.random.normal(
                0,
                300
            )

            rows.append([
                date,
                sku,
                region,
                temp,
                promo,
                event,
                int(sales)
            ])

df = pd.DataFrame(
    rows,
    columns=[
        "date",
        "sku",
        "region",
        "temp",
        "promo",
        "event",
        "sales"
    ]
)

df.to_csv(
    "Hack/data/sales.csv",
    index=False
)

print(df.head())