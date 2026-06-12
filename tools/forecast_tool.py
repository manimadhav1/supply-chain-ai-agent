import pandas as pd
import joblib

# Load artifacts once
model = joblib.load("models/forecast_model.pkl")
sku_encoder = joblib.load("models/sku_encoder.pkl")
region_encoder = joblib.load("models/region_encoder.pkl")
channel_encoder = joblib.load("models/channel_encoder.pkl")
category_encoder = joblib.load("models/category_encoder.pkl")
sku_category_map = joblib.load("models/sku_category_map.pkl")
metadata = joblib.load("models/forecast_model_metadata.pkl")


def encode_or_raise(encoder, value, name):
    value = str(value).strip()
    if value not in encoder.classes_:
        raise ValueError(f"Unknown {name}: {value}. Re-train model with this category or choose a known value.")
    return int(encoder.transform([value])[0])


def forecast_tool(
    sku: str,
    region: str,
    temp: int,
    promo: int,
    event: int,
    month: int,
    dayofweek: int,
    channel: str
):
    sku = str(sku).strip()
    region = str(region).strip()
    channel = str(channel).strip() if channel is not None else ""
    if channel == "":
        channel = channel_encoder.classes_[0]

    category = sku_category_map.get(sku, "Unknown")

    sku_encoded = encode_or_raise(sku_encoder, sku, "sku")
    region_encoded = encode_or_raise(region_encoder, region, "region")
    try:
        channel_encoded = encode_or_raise(channel_encoder, channel, "channel")
    except ValueError:
        channel_encoded = int(channel_encoder.transform([channel_encoder.classes_[0]])[0])
    category_encoded = encode_or_raise(category_encoder, category, "category")

    unit_price = metadata.get("median_unit_price", 0.0)
    discount_pct = metadata.get("median_discount_pct", 0.0)
    is_weekend = 1 if int(dayofweek) in [5, 6] else 0

    features = pd.DataFrame([{
        "month": int(month),
        "dayofweek": int(dayofweek),
        "is_weekend": is_weekend,
        "unit_price": unit_price,
        "discount_pct": discount_pct,
        "sku_encoded": sku_encoded,
        "region_encoded": region_encoded,
        "channel_encoded": channel_encoded,
        "category_encoded": category_encoded
    }])

    prediction = model.predict(features)[0]
    return round(float(prediction), 2)