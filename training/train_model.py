import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

df = pd.read_csv(
    "data/sales.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

df["month"] = df["date"].dt.month
df["dayofweek"] = df["date"].dt.dayofweek

sku_encoder = LabelEncoder()
region_encoder = LabelEncoder()

df["sku_encoded"] = (
    sku_encoder.fit_transform(df["sku"])
)

df["region_encoded"] = (
    region_encoder.fit_transform(
        df["region"]
    )
)

features = [
    "temp",
    "promo",
    "event",
    "month",
    "dayofweek",
    "sku_encoded",
    "region_encoded"
]

X = df[features]

y = df["sales"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6
)

model.fit(
    X_train,
    y_train
)

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

print("Model saved")