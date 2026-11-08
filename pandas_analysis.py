import pandas as pd

# Load cleaned data
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Keep only completed orders
completed_orders = orders[
    orders["status"] == "completed"
].copy()

print("Data loaded successfully!")
print("Customers:", len(customers))
print("Products:", len(products))
print("Completed orders:", len(completed_orders))

# Rebuild Q2 using pandas pivot_table
orders_with_details = completed_orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

orders_with_details = orders_with_details.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

revenue_pivot = pd.pivot_table(
    orders_with_details,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True
)

print("\nRevenue by city and category:")
print(revenue_pivot)

print(
    "\nPandas grand total:",
    round(revenue_pivot.loc["All", "All"], 2)
)

# City-wise completed-order statistics
city_stats = (
    completed_orders
    .merge(
        customers[["customer_id", "city"]],
        on="customer_id",
        how="left"
    )
    .groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
        order_count=("order_id", "count")
    )
    .reset_index()
)

print("\nCity-wise spending statistics:")
print(city_stats.round(2).to_string(index=False))

# Overall completed-order spending distribution
amounts = completed_orders["order_amount"]

q1 = amounts.quantile(0.25)
q3 = amounts.quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + (1.5 * iqr)

print("\nOVERALL SPENDING DISTRIBUTION")
print("Mean:", round(amounts.mean(), 2))
print("Median:", round(amounts.median(), 2))
print("Skewness:", round(amounts.skew(), 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("IQR:", round(iqr, 2))
print("Upper fence:", round(upper_fence, 2))
print("Upper outliers:", (amounts > upper_fence).sum())
print("Lower outliers:", (amounts < q1 - 1.5 * iqr).sum())

# Correlation between completed-order numerical columns
correlation = completed_orders[
    [
        "order_amount",
        "discount_pct",
        "quantity",
        "delivery_minutes"
    ]
].corr()

print("\nCORRELATION MATRIX")
print(correlation.round(3))

# Find the strongest correlation, excluding self-correlations
correlation_pairs = correlation.abs().where(
    ~pd.DataFrame(
        [[i == j for j in range(len(correlation.columns))]
         for i in range(len(correlation.columns))],
        index=correlation.index,
        columns=correlation.columns
    )
)

strongest_pair = correlation_pairs.stack().idxmax()

print("\nStrongest absolute correlation:")
print(strongest_pair)
print("Correlation value:", round(
    correlation.loc[strongest_pair[0], strongest_pair[1]], 3
))

# Save analysis summary
summary = {
    "total_customers": len(customers),
    "total_products": len(products),
    "completed_orders": len(completed_orders),
    "total_revenue": round(completed_orders["order_amount"].sum(), 2),
    "mean_order_amount": round(amounts.mean(), 2),
    "median_order_amount": round(amounts.median(), 2),
    "skewness": round(amounts.skew(), 2),
    "upper_outliers": int((amounts > upper_fence).sum())
}

pd.Series(summary).to_csv(
    "outputs/part_a_summary.csv",
    header=["value"]
)

print("\nPart A summary saved successfully!")