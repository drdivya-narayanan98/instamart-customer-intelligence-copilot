import pandas as pd

# Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

# Convert date columns
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Define prediction cutoff
cutoff_date = pd.Timestamp("2025-09-01")

print("Data loaded successfully!")
print("Customers:", len(customers))
print("Orders:", len(orders))
print("Cutoff date:", cutoff_date)

# Keep orders before the cutoff date
pre_cutoff_orders = orders[
    orders["order_ts"] < cutoff_date
].copy()

# Identify customers with at least one completed order
eligible_customers = pre_cutoff_orders[
    pre_cutoff_orders["status"] == "completed"
]["customer_id"].unique()

# Create the eligible customer cohort
cohort = customers[
    customers["customer_id"].isin(eligible_customers)
].copy()

print("\nCOHORT ANALYSIS")
print("Eligible customers:", len(cohort))
print("Orders before cutoff:", len(pre_cutoff_orders))
print(
    "Completed orders before cutoff:",
    (pre_cutoff_orders["status"] == "completed").sum()
)

# Identify completed orders during September
september_orders = orders[
    (orders["order_ts"] >= cutoff_date) &
    (orders["order_ts"] < pd.Timestamp("2025-10-01")) &
    (orders["status"] == "completed")
]

# Customers who purchased during September
active_in_september = september_orders[
    "customer_id"
].unique()

# Create the lapse label
cohort["lapsed"] = (
    ~cohort["customer_id"].isin(active_in_september)
).astype(int)

print("\nLAPSE LABEL DISTRIBUTION")
print(cohort["lapsed"].value_counts().sort_index())

print("\nLapsed customers:", cohort["lapsed"].sum())
print(
    "Lapsed percentage:",
    round(cohort["lapsed"].mean() * 100, 2),
    "%"
)

# Completed orders before the cutoff
completed_pre = pre_cutoff_orders[
    pre_cutoff_orders["status"] == "completed"
].copy()

# Calculate historical RFM features
rfm_features = (
    completed_pre
    .groupby("customer_id")
    .agg(
        last_order_date=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    )
    .reset_index()
)

# Calculate recency using the cutoff date
rfm_features["recency"] = (
    cutoff_date - rfm_features["last_order_date"]
).dt.days

# Remove intermediate date column
rfm_features = rfm_features.drop(
    columns=["last_order_date"]
)

print("\nPRE-CUTOFF RFM FEATURES")
print(rfm_features.head())
print("\nRFM feature shape:", rfm_features.shape)

# Load cleaned product data
products = pd.read_csv("data/clean/products.csv")

# Calculate order behaviour before the cutoff
order_features = (
    pre_cutoff_orders
    .groupby("customer_id")
    .agg(
        avg_discount=("discount_pct", "mean"),
        total_orders=("order_id", "nunique"),
        cancelled_orders=(
            "status",
            lambda x: (x == "cancelled").sum()
        )
    )
    .reset_index()
)

# Calculate cancellation rate
order_features["cancellation_rate"] = (
    order_features["cancelled_orders"] /
    order_features["total_orders"]
)

# Calculate average delivery time for completed orders
delivery_features = (
    completed_pre
    .groupby("customer_id")
    .agg(
        avg_delivery_minutes=("delivery_minutes", "mean")
    )
    .reset_index()
)

# Find each customer's favourite product category
customer_products = completed_pre.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

favourite_category = (
    customer_products
    .groupby("customer_id")["category"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index(name="favourite_category")
)

# Combine all customer features
customer_features = (
    rfm_features
    .merge(order_features, on="customer_id")
    .merge(delivery_features, on="customer_id")
    .merge(favourite_category, on="customer_id")
    .merge(
        customers[
            ["customer_id", "city", "age", "gender", "signup_date"]
        ],
        on="customer_id"
    )
)

# Calculate customer tenure
customer_features["tenure_days"] = (
    cutoff_date - customer_features["signup_date"]
).dt.days

customer_features = customer_features.drop(
    columns=["signup_date", "total_orders", "cancelled_orders"]
)

print("\nCOMPLETE CUSTOMER FEATURES")
print(customer_features.head().to_string(index=False))
print("\nFeature dataset shape:", customer_features.shape)
print("\nMissing values:")
print(customer_features.isna().sum())

# Combine customer features with lapse labels
model_data = customer_features.merge(
    cohort[["customer_id", "lapsed"]],
    on="customer_id",
    how="inner"
)

# Separate features (X) and target (y)
X = model_data.drop(
    columns=["customer_id", "lapsed"]
)

y = model_data["lapsed"]

print("\nMODELLING DATASET")
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeature columns:")
print(X.columns.tolist())

print("\nMissing feature values:", X.isna().sum().sum())

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTRAIN-TEST SPLIT")
print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Identify numerical and categorical columns
numeric_features = [
    "frequency",
    "monetary",
    "recency",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery_minutes",
    "age",
    "tenure_days"
]

categorical_features = [
    "favourite_category",
    "city",
    "gender"
]

# Create preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(
            drop="first",
            handle_unknown="ignore"
        ), categorical_features)
    ]
)

print("\nPREPROCESSING CONFIGURATION")
print("Numerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))
print("Encoder: OneHotEncoder with drop='first'")
print("Scaler: StandardScaler")

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression pipeline
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

print("\nLOGISTIC REGRESSION PIPELINE CREATED!")

from sklearn.model_selection import GridSearchCV

# Define hyperparameter values
logistic_params = {
    "classifier__C": [0.01, 0.1, 1, 10, 100]
}

# Perform 5-fold cross-validation
logistic_grid = GridSearchCV(
    estimator=logistic_pipeline,
    param_grid=logistic_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train and tune using training data only
logistic_grid.fit(X_train, y_train)

print("\nLOGISTIC REGRESSION RESULTS")
print("Best parameters:", logistic_grid.best_params_)
print("Best CV F1 score:", round(logistic_grid.best_score_, 3))


from sklearn.ensemble import RandomForestClassifier

# Create Random Forest pipeline
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ]
)

# Define hyperparameter values
rf_params = {
    "classifier__max_depth": [3, 5, 8],
    "classifier__min_samples_leaf": [5, 10, 20]
}

# Perform 5-fold cross-validation
rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_params,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

# Train and tune using training data only
rf_grid.fit(X_train, y_train)

print("\nRANDOM FOREST RESULTS")
print("Best parameters:", rf_grid.best_params_)
print("Best CV F1 score:", round(rf_grid.best_score_, 3))

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# Get best tuned models
best_logistic = logistic_grid.best_estimator_
best_rf = rf_grid.best_estimator_

# Generate test predictions
logistic_predictions = best_logistic.predict(X_test)
rf_predictions = best_rf.predict(X_test)

# Evaluate Logistic Regression
print("\nLOGISTIC REGRESSION TEST RESULTS")
print("Accuracy:", round(
    accuracy_score(y_test, logistic_predictions), 3
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))
print("Classification Report:")
print(classification_report(
    y_test,
    logistic_predictions,
    zero_division=0
))

# Evaluate Random Forest
print("\nRANDOM FOREST TEST RESULTS")
print("Accuracy:", round(
    accuracy_score(y_test, rf_predictions), 3
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))
print("Classification Report:")
print(classification_report(
    y_test,
    rf_predictions,
    zero_division=0
))

from sklearn.dummy import DummyClassifier

# Create a simple baseline model
baseline = DummyClassifier(strategy="most_frequent")

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

print("\nBASELINE MODEL RESULTS")
print("Accuracy:", round(
    accuracy_score(y_test, baseline_predictions), 3
))
print("Confusion Matrix:")
print(confusion_matrix(y_test, baseline_predictions))
print("Classification Report:")
print(classification_report(
    y_test,
    baseline_predictions,
    zero_division=0
))


from sklearn.utils import resample
from sklearn.utils import shuffle

# Combine training features and target
train_data = X_train.copy()
train_data["lapsed"] = y_train

# Separate active and lapsed customers
active_customers = train_data[
    train_data["lapsed"] == 0
]

lapsed_customers = train_data[
    train_data["lapsed"] == 1
]

# Oversample lapsed customers
lapsed_oversampled = resample(
    lapsed_customers,
    replace=True,
    n_samples=len(active_customers),
    random_state=42
)

# Combine both classes
balanced_train = pd.concat([
    active_customers,
    lapsed_oversampled
])

# Shuffle the balanced data
balanced_train = shuffle(
    balanced_train,
    random_state=42
)

# Separate features and target
X_train_balanced = balanced_train.drop(
    columns=["lapsed"]
)

y_train_balanced = balanced_train["lapsed"]

print("\nBALANCED TRAINING DATA")
print(y_train_balanced.value_counts())

# Retrain the best Logistic Regression model
balanced_logistic = logistic_grid.best_estimator_

balanced_logistic.fit(
    X_train_balanced,
    y_train_balanced
)

# Predict on the original test data
balanced_predictions = balanced_logistic.predict(X_test)

print("\nBALANCED LOGISTIC REGRESSION RESULTS")

print("Accuracy:", round(
    accuracy_score(y_test, balanced_predictions), 3
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, balanced_predictions))

print("Classification Report:")
print(classification_report(
    y_test,
    balanced_predictions,
    zero_division=0
))

# Extract feature importance from the tuned Random Forest
best_rf = rf_grid.best_estimator_

# Get feature names after preprocessing
feature_names = (
    best_rf
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Get feature importance scores
importances = (
    best_rf
    .named_steps["classifier"]
    .feature_importances_
)

# Create a feature importance table
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

# Sort by importance
importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("\nTOP 5 RANDOM FOREST FEATURES")
print(importance_df.head(5).to_string(index=False))

# Extract Logistic Regression coefficients
logistic_coefficients = (
    best_logistic
    .named_steps["classifier"]
    .coef_[0]
)

# Get feature names after preprocessing
logistic_feature_names = (
    best_logistic
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

# Create coefficient table
coefficient_df = pd.DataFrame({
    "feature": logistic_feature_names,
    "coefficient": logistic_coefficients
})

# Sort by absolute coefficient magnitude
coefficient_df["absolute_coefficient"] = (
    coefficient_df["coefficient"].abs()
)

coefficient_df = coefficient_df.sort_values(
    by="absolute_coefficient",
    ascending=False
)

print("\nTOP 3 LOGISTIC REGRESSION COEFFICIENTS")
print(
    coefficient_df[
        ["feature", "coefficient"]
    ].head(3).to_string(index=False)
)

import joblib

# Select the tuned Random Forest model
selected_model = best_rf

# Save the trained model
joblib.dump(
    selected_model,
    "outputs/lapse_model.joblib"
)

print("\nMODEL SAVED SUCCESSFULLY!")

# Reload the saved model
loaded_model = joblib.load(
    "outputs/lapse_model.joblib"
)

# Generate predictions using the reloaded model
loaded_predictions = loaded_model.predict(X_test)

# Verify predictions match
predictions_match = (
    loaded_predictions == rf_predictions
).all()

print("\nMODEL RELOAD VERIFICATION")
print("Original predictions:", len(rf_predictions))
print("Reloaded predictions:", len(loaded_predictions))
print("Predictions identical:", predictions_match)

# Get lapse probabilities from the selected model
lapse_probabilities = loaded_model.predict_proba(X_test)[:, 1]

# Recover customer IDs for the test rows
test_customer_ids = model_data.loc[
    X_test.index, "customer_id"
].values

# Create prediction results
retention_list = pd.DataFrame({
    "customer_id": test_customer_ids,
    "lapse_probability": lapse_probabilities,
    "actually_lapsed": y_test.values
})

# Add Premium segment information
segments = pd.read_csv("outputs/customer_segments.csv")

retention_list = retention_list.merge(
    segments[["customer_id", "is_premium"]],
    on="customer_id",
    how="left"
)

# Keep Premium customers only
retention_list = retention_list[
    retention_list["is_premium"] == True
].copy()

# Rank by lapse probability
retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False
).head(10)

# Keep required output columns
retention_list = retention_list[
    [
        "customer_id",
        "lapse_probability",
        "actually_lapsed"
    ]
]

# Save retention list
retention_list.to_csv(
    "outputs/retention_list.csv",
    index=False
)

print("\nTOP 10 PREMIUM CUSTOMERS FOR RETENTION")
print(retention_list.to_string(index=False))

print("\nRetention list saved successfully!")
print(
    "Actually lapsed customers in list:",
    retention_list["actually_lapsed"].sum()
)