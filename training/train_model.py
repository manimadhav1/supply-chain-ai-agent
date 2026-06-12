import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load sales transactions from the uploaded workbook
sales_df = pd.read_excel(
    "data/FMCG_DemandForecasting_Dataset.xlsx",
    sheet_name="Sales_Transactions",
    header=1
)

sales_df = sales_df.rename(columns={
    "SKU ID": "sku",
    "Region": "region",
    "Channel": "channel",
    "Qty Sold": "qty_sold",
    "Unit Price (₹)": "unit_price",
    "Discount (%)": "discount_pct",
    "Net Price (₹)": "net_price",
    "Revenue (₹)": "revenue",
    "COGS (₹)": "cogs",
    "Date": "date",
    "Category": "category"
})

sales_df = sales_df.dropna(subset=["sku", "region", "date", "qty_sold"])

sales_df["date"] = pd.to_datetime(sales_df["date"], errors="coerce")
sales_df = sales_df.dropna(subset=["date"])

sales_df["month"] = sales_df["date"].dt.month
sales_df["dayofweek"] = sales_df["date"].dt.dayofweek
sales_df["is_weekend"] = sales_df["dayofweek"].isin([5, 6]).astype(int)

sales_df["channel"] = sales_df["channel"].astype(str).str.strip().fillna("Unknown")

if "category" in sales_df.columns:
    sales_df["category"] = sales_df["category"].astype(str).str.strip().fillna("Unknown")
else:
    sales_df["category"] = "Unknown"

sales_df["unit_price"] = sales_df["unit_price"].fillna(sales_df["unit_price"].median()).astype(float)
sales_df["discount_pct"] = sales_df["discount_pct"].fillna(0).astype(float)
sales_df["qty_sold"] = sales_df["qty_sold"].astype(float)

sku_encoder = LabelEncoder()
region_encoder = LabelEncoder()
channel_encoder = LabelEncoder()
category_encoder = LabelEncoder()

sales_df["sku_encoded"] = sku_encoder.fit_transform(sales_df["sku"])
sales_df["region_encoded"] = region_encoder.fit_transform(sales_df["region"])
sales_df["channel_encoded"] = channel_encoder.fit_transform(sales_df["channel"])
sales_df["category_encoded"] = category_encoder.fit_transform(sales_df["category"])

features = [
    "month",
    "dayofweek",
    "is_weekend",
    "unit_price",
    "discount_pct",
    "sku_encoded",
    "region_encoded",
    "channel_encoded",
    "category_encoded"
]

X = sales_df[features]
y = sales_df["qty_sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Training complete. MAE={mae:.2f}, RMSE={rmse:.2f}")

joblib.dump(model, "models/forecast_model.pkl")
joblib.dump(sku_encoder, "models/sku_encoder.pkl")
joblib.dump(region_encoder, "models/region_encoder.pkl")
joblib.dump(channel_encoder, "models/channel_encoder.pkl")
joblib.dump(category_encoder, "models/category_encoder.pkl")

sku_category_map = sales_df.groupby("sku")["category"].first().to_dict()
joblib.dump(sku_category_map, "models/sku_category_map.pkl")

metadata = {
    "median_unit_price": float(sales_df["unit_price"].median()),
    "median_discount_pct": float(sales_df["discount_pct"].median())
}
joblib.dump(metadata, "models/forecast_model_metadata.pkl")

print("Saved model, encoders, and metadata to models/")