import pandas as pd
import numpy as np

np.random.seed(42)

master = pd.read_csv("data/product_master.csv")

regions = ["Bangalore", "Chennai", "Mumbai", "Hyderabad"]

region_mult = {"Mumbai": 1.30, "Bangalore": 1.15, "Hyderabad": 1.00, "Chennai": 0.90}

# Base daily demand in units (regional FMCG distributor serving ~100 outlets)
sku_base_demand = {
    "SKU-001": 1200,  # Colgate - daily use, market leader
    "SKU-002": 550,   # Surf Excel - monthly bulk, lower daily turns
    "SKU-003": 2500,  # Maggi - very high velocity, impulse + meal
    "SKU-004": 3500,  # Lays - impulse snack, highest velocity
    "SKU-005": 750,   # Tropicana - premium, moderate volume
    "SKU-006": 450,   # Horlicks - health segment, niche
    "SKU-007": 900,   # Dettol - hygiene essential, steady
    "SKU-008": 2000,  # Britannia Good Day - popular biscuit
    "SKU-009": 680,   # Amul Butter - dairy essential
    "SKU-010": 380,   # Red Bull - premium energy, low volume
    "SKU-011": 780,   # Dove Soap - personal care
    "SKU-012": 3000,  # Parle-G - mass market, cheapest biscuit
    "SKU-013": 2800,  # Kurkure - very popular snack
    "SKU-014": 950,   # Tata Salt - kitchen staple
    "SKU-015": 650,   # Clinic Plus - hair care
    "SKU-016": 580,   # Vim - household essential
    "SKU-017": 480,   # Sunflower Oil - bulk grocery
    "SKU-018": 320,   # Fortune Rice - bulk, low daily turns
    "SKU-019": 1400,  # KitKat - popular impulse chocolate
    "SKU-020": 190,   # Gillette - premium, very low frequency
}

# Seasonal multiplier by category and month (Jan=1 ... Dec=12)
# Based on actual Indian FMCG consumption patterns
category_seasonal = {
    "Beverages": {
        1: 0.80, 2: 0.85, 3: 1.00, 4: 1.25, 5: 1.50, 6: 1.45,
        7: 0.95, 8: 0.90, 9: 0.95, 10: 1.00, 11: 0.90, 12: 0.80
    },
    "Snacks": {
        1: 0.95, 2: 0.90, 3: 1.00, 4: 1.05, 5: 1.10, 6: 1.05,
        7: 1.05, 8: 1.05, 9: 1.10, 10: 1.25, 11: 1.30, 12: 1.10
    },
    "Food": {
        1: 1.10, 2: 1.00, 3: 0.95, 4: 0.90, 5: 0.88, 6: 1.05,
        7: 1.35, 8: 1.40, 9: 1.25, 10: 1.10, 11: 1.05, 12: 1.15
    },
    "Dairy": {
        1: 1.15, 2: 1.10, 3: 1.00, 4: 0.95, 5: 0.88, 6: 0.88,
        7: 0.92, 8: 0.92, 9: 0.95, 10: 1.05, 11: 1.15, 12: 1.20
    },
    "Health & Nutrition": {
        1: 1.35, 2: 1.25, 3: 1.00, 4: 0.85, 5: 0.78, 6: 0.80,
        7: 0.88, 8: 0.88, 9: 0.92, 10: 1.05, 11: 1.20, 12: 1.38
    },
    "Personal Care": {
        1: 0.95, 2: 0.95, 3: 1.00, 4: 1.02, 5: 1.05, 6: 1.05,
        7: 1.00, 8: 1.00, 9: 1.00, 10: 1.05, 11: 1.12, 12: 1.05
    },
    "Home Care": {
        1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00, 5: 1.00, 6: 1.08,
        7: 1.12, 8: 1.12, 9: 1.05, 10: 1.00, 11: 1.00, 12: 1.00
    },
    "Grocery": {
        1: 1.05, 2: 1.00, 3: 1.00, 4: 1.00, 5: 1.00, 6: 1.00,
        7: 1.00, 8: 1.00, 9: 1.00, 10: 1.08, 11: 1.12, 12: 1.10
    },
}

# Day of week multipliers (Mon=0 ... Sun=6)
category_dow_mult = {
    "Beverages":          [0.85, 0.88, 0.90, 0.95, 1.10, 1.22, 1.10],
    "Snacks":             [0.85, 0.88, 0.90, 0.95, 1.10, 1.28, 1.16],
    "Food":               [0.98, 1.00, 1.00, 1.00, 1.05, 1.08, 1.02],
    "Dairy":              [1.00, 1.00, 1.00, 1.00, 1.05, 1.12, 1.05],
    "Health & Nutrition": [1.08, 1.02, 1.00, 1.00, 1.00, 0.95, 0.95],
    "Personal Care":      [0.95, 1.00, 1.00, 1.00, 1.00, 1.12, 1.05],
    "Home Care":          [0.95, 1.00, 1.00, 1.00, 1.00, 1.12, 1.05],
    "Grocery":            [1.08, 1.02, 1.00, 1.00, 1.00, 1.12, 1.05],
}

# Temperature sensitivity per °C above 30 (category-specific)
temp_sensitivity = {
    "Beverages":          0.028,
    "Snacks":             0.008,
    "Food":              -0.006,
    "Dairy":             -0.010,
    "Health & Nutrition":-0.005,
    "Personal Care":      0.004,
    "Home Care":          0.000,
    "Grocery":            0.000,
}

# Event type effects: [festival, sports, national_holiday, wedding_season]
event_effects = {
    "Beverages":          [0.45, 0.65, 0.22, 0.12],
    "Snacks":             [0.55, 0.52, 0.25, 0.18],
    "Food":               [0.30, 0.22, 0.20, 0.12],
    "Dairy":              [0.22, 0.08, 0.12, 0.35],
    "Health & Nutrition": [0.10, 0.05, 0.10, 0.05],
    "Personal Care":      [0.25, 0.05, 0.15, 0.40],
    "Home Care":          [0.30, 0.05, 0.20, 0.32],
    "Grocery":            [0.38, 0.10, 0.28, 0.45],
}

# Promo type multipliers: 0=none, 1=discount_low, 2=discount_high, 3=bogo, 4=bundle
promo_base_mult = {0: 1.00, 1: 1.22, 2: 1.42, 3: 1.75, 4: 1.32}

rows = []

dates = pd.date_range(start="2023-01-01", end="2025-12-31")  # 3 years for rich data

for _, sku_row in master.iterrows():
    sku_id = sku_row["sku_id"]
    category = sku_row["category"]
    unit_cost = float(sku_row["unit_cost"])
    shelf_life = float(sku_row["shelf_life"])

    base = sku_base_demand.get(sku_id, 500)
    seasonal_map = category_seasonal[category]
    dow_map = category_dow_mult[category]
    temp_sens = temp_sensitivity[category]
    evt_effects = event_effects[category]

    for region in regions:
        r_mult = region_mult[region]

        for date in dates:
            month = date.month
            dow = date.weekday()
            is_weekend = int(dow >= 5)

            # Temperature — regional variation
            temp_base = {
                "Mumbai": 30, "Bangalore": 25, "Chennai": 32, "Hyderabad": 28
            }[region]
            season_temp_offset = {
                1: -3, 2: -2, 3: 2, 4: 7, 5: 10, 6: 8,
                7: 4, 8: 4, 9: 3, 10: 0, 11: -2, 12: -4
            }[month]
            temperature = int(np.clip(
                temp_base + season_temp_offset + np.random.normal(0, 2),
                18, 46
            ))

            # Promo — realistic frequency by category
            promo_freq = {
                "Beverages": 0.12, "Snacks": 0.15, "Food": 0.10,
                "Dairy": 0.06, "Health & Nutrition": 0.08,
                "Personal Care": 0.10, "Home Care": 0.10, "Grocery": 0.08
            }[category]

            if np.random.random() < promo_freq:
                promo_type = np.random.choice([1, 2, 3, 4], p=[0.45, 0.30, 0.15, 0.10])
            else:
                promo_type = 0

            if promo_type in [1, 2]:
                promo_discount_pct = np.random.choice([5, 10, 15, 20, 25, 30], p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
            else:
                promo_discount_pct = 0

            # Event — realistic frequencies for India
            event_rand = np.random.random()
            if month in [10, 11] and event_rand < 0.12:   # Diwali/festive season
                event_type = 1
            elif month in [3, 4] and event_rand < 0.08:   # IPL
                event_type = 2
            elif event_rand < 0.04:                        # National holiday
                event_type = 3
            elif month in [11, 12, 1, 2] and event_rand < 0.06:  # Wedding season
                event_type = 4
            else:
                event_type = 0

            # Competitor stockout (random, 3% of days)
            is_competitor_stockout = int(np.random.random() < 0.03)

            # ---- DEMAND CALCULATION ----

            seasonal_mult = seasonal_map[month]
            dow_mult = dow_map[dow]
            temp_effect = 1 + temp_sens * max(0, temperature - 30)
            promo_mult = promo_base_mult[promo_type]
            if promo_type in [1, 2] and promo_discount_pct > 10:
                promo_mult += 0.05 * ((promo_discount_pct - 10) / 10)
            event_mult = 1 + evt_effects[event_type - 1] if event_type > 0 else 1.0
            competitor_mult = 1.20 if is_competitor_stockout else 1.0

            sales = (
                base
                * r_mult
                * seasonal_mult
                * dow_mult
                * temp_effect
                * promo_mult
                * event_mult
                * competitor_mult
            )

            # Realistic noise (heteroskedastic — higher volume = more variance)
            noise = np.random.normal(0, sales * 0.07)
            sales = max(10, int(sales + noise))

            rows.append({
                "date": date,
                "sku_id": sku_id,
                "product_name": sku_row["product_name"],
                "category": category,
                "region": region,
                "month": month,
                "dayofweek": dow,
                "is_weekend": is_weekend,
                "temperature": temperature,
                "promo_type": promo_type,
                "promo_discount_pct": promo_discount_pct,
                "event_type": event_type,
                "is_competitor_stockout": is_competitor_stockout,
                "unit_cost": unit_cost,
                "shelf_life_days": shelf_life,
                "sales": sales,
            })

df = pd.DataFrame(rows)
df.to_csv("data/sales_v2.csv", index=False)

print(f"Generated {len(df):,} rows")
print(f"SKUs: {df['sku_id'].nunique()}, Regions: {df['region'].nunique()}")
print(f"\nSample demand ranges:")
sample = df.groupby(["sku_id", "product_name"])["sales"].agg(["mean", "min", "max"]).round(0)
print(sample.to_string())
