import sqlite3
import pandas as pd

# Load cleaned data
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

completed = orders[orders["status"] == "completed"]
cancelled = orders[orders["status"] == "cancelled"]

# Check 1: Customer count
assert len(customers) == 1200

# Check 2: Product count
assert len(products) == 40

# Check 3: Customer ages have no missing values
assert customers["age"].isna().sum() == 0

# Check 4: Exactly six standardised cities
assert customers["city"].nunique() == 6

# Check 5: Gender categories are standardised
assert set(customers["gender"].unique()) == {
    "male", "female", "unknown"
}

# Check 6: Every order has a unique order ID
assert len(orders) == 12309
assert orders["order_id"].nunique() == 12309

# Check 7: No pending orders remain
assert "pending" not in orders["status"].str.lower().unique()

# Check 8: No missing order amounts
assert orders["order_amount"].isna().sum() == 0

# Check 9: Completed orders have delivery times
assert completed["delivery_minutes"].isna().sum() == 0

# Check 10: Cancelled orders have missing delivery times
assert cancelled["delivery_minutes"].isna().sum() == 1199

# Check 11: Completed order revenue
assert abs(completed["order_amount"].sum() - 2998668.80) < 0.01

# Check 12: SQLite database contains all three tables
with sqlite3.connect("data/instamart.db") as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    assert {"customers", "products", "orders"}.issubset(tables)

    assert connection.execute(
        "SELECT COUNT(*) FROM customers"
    ).fetchone()[0] == 1200

    assert connection.execute(
        "SELECT COUNT(*) FROM products"
    ).fetchone()[0] == 40

    assert connection.execute(
        "SELECT COUNT(*) FROM orders"
    ).fetchone()[0] == 12309

print("ALL CHECKPOINTS PASSED")