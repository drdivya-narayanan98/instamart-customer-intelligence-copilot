
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

for file_path in sorted(RAW_DIR.glob("*.csv")):
    print("\n" + "=" * 65)
    print(f"FILE: {file_path.name}")
    print("=" * 65)

    if file_path.name == "tickets.csv":
        df = pd.read_csv(file_path, keep_default_na=False)
    else:
        df = pd.read_csv(file_path)

    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nColumn data types:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()

    # Count empty strings as missing in tickets.csv
    if file_path.name == "tickets.csv":
        blank_counts = df.eq("").sum()
        missing = missing + blank_counts

    print("\nMissing values:")
    for column in df.columns:
        count = int(missing[column])
        percentage = count / len(df) * 100 if len(df) else 0
        print(f"{column}: {count} ({percentage:.2f}%)")
