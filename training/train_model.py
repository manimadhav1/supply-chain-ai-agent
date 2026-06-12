import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv("data/sales_v2.csv")

sku_enc    = LabelEncoder()
cat_enc    = LabelEncoder()
region_enc = LabelEncoder()

df["sku_encoded"]      = sku_enc.fit_transform(df["sku_id"])
df["category_encoded"] = cat_enc.fit_transform(df["category"])
df["region_encoded"]   = region_enc.fit_transform(df["region"])

FEATURES = [
    "sku_encoded", "category_encoded", "region_encoded",
    "month", "dayofweek", "is_weekend",
    "temperature",
    "promo_type", "promo_discount_pct",
    "event_type", "is_competitor_stockout",
    "unit_cost", "shelf_life_days",
]
TARGET = "sales"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=30,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

y_pred = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
print(f"\n✅ Test MAPE: {mape:.2f}%")

fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importance:")
print(fi.round(4).to_string())

joblib.dump(model,    "models/forecast_model.pkl")
joblib.dump(sku_enc,  "models/sku_encoder.pkl")
joblib.dump(cat_enc,  "models/category_encoder.pkl")
joblib.dump(region_enc, "models/region_encoder.pkl")
joblib.dump(FEATURES, "models/feature_list.pkl")

print("\n✅ Model and encoders saved to models/")
