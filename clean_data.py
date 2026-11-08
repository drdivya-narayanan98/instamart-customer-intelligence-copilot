
import pandas as pd
from pathlib import Path

# Create the clean-data folder if it does not exist
Path("data/clean").mkdir(parents=True, exist_ok=True)

# Load raw customers
customers = pd.read_csv("data/raw/customers.csv")

# 1. Standardise city names
customers["city"] = (
    customers["city"]
    .str.strip()
    .str.title()
)

# 2. Standardise gender values
customers["gender"] = (
    customers["gender"]
    .fillna("")
    .str.strip()
    .str.lower()
    .map({
        "m": "male",
        "male": "male",
        "f": "female",
        "female": "female"
    })
    .fillna("unknown")
)

# 3. Treat ages outside 10-100 as missing
customers.loc[
    ~customers["age"].between(10, 100),
    "age"
] = pd.NA

# 4. Fill missing ages using the median age of each city
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

# 5. Convert signup_date into datetime
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"]
)

# Save clean customers
customers.to_csv(
    "data/clean/customers.csv",
    index=False
)

print("Customer cleaning completed!")
print("\nCity counts:")
print(customers["city"].value_counts())

print("\nGender counts:")
print(customers["gender"].value_counts())

print("\nMissing ages:", customers["age"].isna().sum())
print(
    "Age range:",
    customers["age"].min(),
    "to",
    customers["age"].max()
)

# -----------------------------------------
# CLEAN ORDERS
# -----------------------------------------

# Load raw orders and products
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

# 1. Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 2. Remove exact duplicate rows
orders = orders.drop_duplicates()

# 3. Keep the latest record for each order_id
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(
    subset="order_id",
    keep="last"
)

# 4. Convert order_amount to numeric
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"],
    errors="coerce"
)

# Add product price and category information
orders = orders.merge(
    products[["product_id", "price_inr", "category"]],
    on="product_id",
    how="left"
)

# Fill missing order amounts using product price,
# quantity and discount
missing_amount = orders["order_amount"].isna()

orders.loc[missing_amount, "order_amount"] = (
    orders.loc[missing_amount, "price_inr"]
    * orders.loc[missing_amount, "quantity"]
    * (
        1 - orders.loc[missing_amount, "discount_pct"] / 100
    )
).round(2)

# 5. Set negative delivery times to missing
orders.loc[
    orders["delivery_minutes"] < 0,
    "delivery_minutes"
] = pd.NA

# Fill missing delivery times for completed orders
completed = orders["status"].str.lower().eq("completed")

category_medians = (
    orders.loc[completed]
    .groupby("category")["delivery_minutes"]
    .transform("median")
)

orders.loc[completed, "delivery_minutes"] = (
    orders.loc[completed, "delivery_minutes"]
    .fillna(category_medians)
)

# Save clean orders and products
orders = orders.drop(columns=["price_inr", "category"])

orders.to_csv(
    "data/clean/orders.csv",
    index=False
)

products.to_csv(
    "data/clean/products.csv",
    index=False
)

print("\nOrders cleaning completed!")