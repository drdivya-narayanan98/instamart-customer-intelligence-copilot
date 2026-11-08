
import pandas as pd

# Load the raw datasets
customers = pd.read_csv("data/raw/customers.csv")
orders = pd.read_csv("data/raw/orders.csv")

# Convert order_amount to numeric.
# Invalid text values become NaN for analysis only.
orders["order_amount_numeric"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)

print("=" * 55)
print("CUSTOMER DATA QUALITY")
print("=" * 55)

print("Missing ages:", customers["age"].isna().sum())

print(
    "Ages outside 10-100:",
    ((customers["age"] < 10) | (customers["age"] > 100)).sum()
)

print("Distinct raw city spellings:", customers["city"].nunique())
print(
    "Distinct nonblank raw gender spellings:",
    customers.loc[customers["gender"].ne(""), "gender"].nunique()
)
print("Blank genders:", customers["gender"].eq("").sum())

print("\n" + "=" * 55)
print("ORDER DATA QUALITY")
print("=" * 55)

print("Exact duplicate rows:", orders.duplicated().sum())

print(
    "Pending orders:",
    orders["status"].str.lower().eq("pending").sum()
)

print("Distinct order IDs:", orders["order_id"].nunique())

print(
    "Nonnumeric order amounts:",
    orders["order_amount_numeric"].isna().sum()
    - orders["order_amount"].isna().sum()
)

print(
    "Negative delivery times:",
    (orders["delivery_minutes"] < 0).sum()
)

print(
    "Blank delivery times:",
    orders["delivery_minutes"].isna().sum()
)
