import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load sales data
df = pd.read_csv("data/sales.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Create derived features
df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

# Encode categorical columns
sku_encoder = LabelEncoder()
region_encoder = LabelEncoder()

df["sku_encoded"] = sku_encoder.fit_transform(
    df["sku"]
)

df["region_encoded"] = region_encoder.fit_transform(
    df["region"]
)

# Features expected by forecast_tool.py
X = df[
    [
        "temp",
        "promo",
        "event",
        "month",
        "dayofweek",
        "sku_encoded",
        "region_encoded"
    ]
]

# Target
y = df["sales"]

# Train model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# Save artifacts
joblib.dump(
    model,
    "models/forecast_model.pkl"
)

joblib.dump(
    sku_encoder,
    "models/sku_encoder.pkl"
)

joblib.dump(
    region_encoder,
    "models/region_encoder.pkl"
)

print("Model training complete.")
print("Artifacts saved to models/")