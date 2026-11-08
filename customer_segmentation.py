import pandas as pd

# Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Keep completed orders only
completed_orders = orders[
    orders["status"] == "completed"
].copy()

# Convert order timestamp to datetime
completed_orders["order_ts"] = pd.to_datetime(
    completed_orders["order_ts"]
)

print("Data loaded successfully!")
print("Total customers:", len(customers))
print("Completed orders:", len(completed_orders))

# RFM snapshot date
snapshot_date = pd.Timestamp("2025-10-01")

# Calculate RFM for each customer
rfm = (
    completed_orders
    .groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

# Recency = days since the last completed order
rfm["recency"] = (
    snapshot_date - rfm["last_order_date"]
).dt.days

# Remove the intermediate date column
rfm = rfm.drop(columns=["last_order_date"])

print("\nRFM analysis completed!")
print(rfm.head(10).to_string(index=False))

print("\nRFM dataset shape:", rfm.shape)

print("\nRFM DATA QUALITY CHECK")
print("Missing values:")
print(rfm.isna().sum())

print("\nMinimum values:")
print(rfm[["recency", "frequency", "monetary"]].min())

print("\nMaximum values:")
print(rfm[["recency", "frequency", "monetary"]].max())

print("\nRFM summary:")
print(rfm[["recency", "frequency", "monetary"]].describe().round(2))

from sklearn.preprocessing import StandardScaler

# Select RFM features
rfm_features = rfm[
    ["recency", "frequency", "monetary"]
]

# Standardise the features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

print("\nRFM scaling completed!")
print("Scaled data shape:", rfm_scaled.shape)
print("Scaled feature means:", rfm_scaled.mean(axis=0).round(3))
print("Scaled feature standard deviations:", rfm_scaled.std(axis=0).round(3))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Compare different numbers of clusters
print("\nK-MEANS COMPARISON")

for k in range(2, 9):
    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(rfm_scaled)

    silhouette = silhouette_score(rfm_scaled, labels)

    print(
        f"K={k}: "
        f"Inertia={model.inertia_:.1f}, "
        f"Silhouette={silhouette:.3f}"
    )

# Train the final K=4 model
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

print("\nK-means clustering completed!")
print("\nNumber of customers in each cluster:")
print(rfm["cluster"].value_counts().sort_index())

# Analyse customer behaviour in each cluster
cluster_summary = (
    rfm.groupby("cluster")
    .agg(
        customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean")
    )
    .round(2)
)

print("\nCUSTOMER CLUSTER SUMMARY")
print(cluster_summary)

# Map cluster numbers to meaningful segment names
segment_names = {
    0: "Regular",
    1: "High Value",
    2: "Loyal",
    3: "At Risk"
}

rfm["segment_id"] = rfm["cluster"]
rfm["segment_name"] = rfm["segment_id"].map(segment_names)
rfm["is_premium"] = rfm["segment_id"] == 1

print("\nCUSTOMER SEGMENT COUNTS")
print(rfm["segment_name"].value_counts())

print("\nSample segmented customers:")
print(
    rfm[
        ["customer_id", "recency", "frequency",
         "monetary", "segment_name"]
    ].head(10).to_string(index=False)
)

# Apply PCA to reduce RFM data to two dimensions
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled)

print("\nPCA ANALYSIS")
print("Explained variance ratio:", pca.explained_variance_ratio_.round(3))
print(
    "Total explained variance:",
    round(pca.explained_variance_ratio_.sum(), 3)
)

# Save customer segmentation results
from pathlib import Path

Path("outputs").mkdir(exist_ok=True)

# Save only the required columns in the correct order
rfm_output = rfm[
    [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "segment_id",
        "segment_name",
        "is_premium"
    ]
]

rfm_output.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

print("\nCustomer segments saved successfully!")
print("File: outputs/customer_segments.csv")