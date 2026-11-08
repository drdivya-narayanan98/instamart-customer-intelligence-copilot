import sqlite3
import pandas as pd
from pathlib import Path

# Database location
Path("data").mkdir(exist_ok=True)
db_path = "data/instamart.db"

# Read cleaned CSV files
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Keep order timestamps in the required text format
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"]
).dt.strftime("%Y-%m-%d %H:%M:%S")

# Create SQLite database and tables
with sqlite3.connect(db_path) as connection:
    customers.to_sql(
        "customers", connection, if_exists="replace", index=False
    )
    products.to_sql(
        "products", connection, if_exists="replace", index=False
    )
    orders.to_sql(
        "orders", connection, if_exists="replace", index=False
    )

print("SQLite database created successfully!")
print("Database:", db_path)