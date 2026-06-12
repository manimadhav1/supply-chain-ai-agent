import pandas as pd
import joblib

# Load artifacts once
model = joblib.load(
    "models/forecast_model.pkl"
)

sku_encoder = joblib.load(
    "models/sku_encoder.pkl"
)

region_encoder = joblib.load(
    "models/region_encoder.pkl"
)


def forecast_tool(
    sku: str,
    region: str,
    temp: int,
    promo: int,
    event: int,
    month: int,
    dayofweek: int
):

    sku_encoded = sku_encoder.transform(
        [sku]
    )[0]

    region_encoded = region_encoder.transform(
        [region]
    )[0]

    features = pd.DataFrame([{
        "temp": temp,
        "promo": promo,
        "event": event,
        "month": month,
        "dayofweek": dayofweek,
        "sku_encoded": sku_encoded,
        "region_encoded": region_encoded
    }])

    prediction = model.predict(
        features
    )[0]

    return round(prediction, 2)