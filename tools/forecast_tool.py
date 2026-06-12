import pandas as pd
import joblib
import os

_MODEL_DIR = "models"

model      = joblib.load(os.path.join(_MODEL_DIR, "forecast_model.pkl"))
sku_enc    = joblib.load(os.path.join(_MODEL_DIR, "sku_encoder.pkl"))
cat_enc    = joblib.load(os.path.join(_MODEL_DIR, "category_encoder.pkl"))
region_enc = joblib.load(os.path.join(_MODEL_DIR, "region_encoder.pkl"))
FEATURES   = joblib.load(os.path.join(_MODEL_DIR, "feature_list.pkl"))

# Product master lookup — needed for category, unit_cost, shelf_life
_master = pd.read_csv("data/product_master.csv").set_index("sku_id")


def forecast_tool(
    sku_id: str,
    region: str,
    month: int,
    dayofweek: int,
    is_weekend: int,
    temperature: int,
    promo_type: int,
    promo_discount_pct: int,
    event_type: int,
    is_competitor_stockout: int,
) -> float:
    row = _master.loc[sku_id]
    category   = row["category"]
    unit_cost  = float(row["unit_cost"])
    shelf_life = float(row["shelf_life"])

    features = pd.DataFrame([{
        "sku_encoded":           sku_enc.transform([sku_id])[0],
        "category_encoded":      cat_enc.transform([category])[0],
        "region_encoded":        region_enc.transform([region])[0],
        "month":                 month,
        "dayofweek":             dayofweek,
        "is_weekend":            is_weekend,
        "temperature":           temperature,
        "promo_type":            promo_type,
        "promo_discount_pct":    promo_discount_pct,
        "event_type":            event_type,
        "is_competitor_stockout": is_competitor_stockout,
        "unit_cost":             unit_cost,
        "shelf_life_days":       shelf_life,
    }])[FEATURES]

    prediction = model.predict(features)[0]
    return round(float(prediction), 2)
