"""
02_preprocessing.py
Netflix Retention Modeling - Data Preprocessing

This script performs:

1. Load prepared dataset
2. Remove unnecessary features
3. Filter outliers
4. Handle missing values
5. Generate churn labels
6. Save the cleaned dataset
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "watch_joined.csv"
OUTPUT_PATH = DATA_DIR / "watch_preprocessed.csv"

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────

df = pd.read_csv(INPUT_PATH, parse_dates=["watch_date"])

print("=" * 50)
print("1. Raw Dataset")
print("=" * 50)

print(f"Total rows    : {len(df):,}")
print(f"Total columns : {df.shape[1]}")

# ─────────────────────────────────────────
# 2. Remove Unnecessary Features
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("2. Dropping Unnecessary Features")
print("=" * 50)

drop_cols = [
    "progress_percentage",   # duplicate of completion_rate
    "watch_ratio",           # derived feature
    "user_rating",           # 79.9% missing values
    "genre_secondary",       # 64.4% missing values
    "session_id"             # identifier only
]

df = df.drop(columns=drop_cols)

print(f"Dropped columns: {drop_cols}")
print(f"Remaining columns: {df.shape[1]}")

# ─────────────────────────────────────────
# 3. Outlier Removal
# ratio = watch_duration / duration
# Remove sessions where ratio >= 3
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("3. Outlier Removal (watch_duration / duration >= 3)")
print("=" * 50)

before = len(df)

df_valid = df.dropna(
    subset=["watch_duration_minutes", "duration_minutes"]
).copy()

df_valid["ratio"] = (
    df_valid["watch_duration_minutes"]
    / df_valid["duration_minutes"]
)

remove_idx = df_valid[df_valid["ratio"] >= 3].index

df = df.drop(index=remove_idx)

df = df.drop(columns=["ratio"], errors="ignore")

after = len(df)

print(f"Rows before filtering : {before:,}")
print(f"Rows after filtering  : {after:,}")
print(f"Removed rows          : {before-after:,}")

# ─────────────────────────────────────────
# 4. Handle Missing Values
# Median Imputation
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("4. Missing Value Imputation (Median)")
print("=" * 50)

for col in ["completion_rate", "watch_duration_minutes"]:

    n_missing = df[col].isnull().sum()

    median_val = df[col].median()

    df[col] = df[col].fillna(median_val)

    print(
        f"{col}: {n_missing:,} missing values "
        f"→ filled with median ({median_val:.2f})"
    )

# ─────────────────────────────────────────
# 5. Generate Churn Label
# 30-day inactivity rule
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("5. Generating Churn Label (30-day inactivity rule)")
print("=" * 50)

reference_date = df["watch_date"].max()

last_watch = (
    df.groupby("user_id")["watch_date"]
    .max()
    .reset_index()
)

last_watch.columns = [
    "user_id",
    "last_watch_date"
]

last_watch["recency_days"] = (
    reference_date - last_watch["last_watch_date"]
).dt.days

last_watch["churned"] = (
    last_watch["recency_days"] >= 30
).astype(int)

df = df.merge(
    last_watch[
        ["user_id", "last_watch_date", "recency_days", "churned"]
    ],
    on="user_id",
    how="left"
)

print(f"Reference date : {reference_date.date()}")

print(
    f"Churned users  : {last_watch['churned'].sum():,} "
    f"({last_watch['churned'].mean()*100:.1f}%)"
)

print(
    f"Retained users : {(1-last_watch['churned']).sum():,} "
    f"({(1-last_watch['churned']).mean()*100:.1f}%)"
)

# ─────────────────────────────────────────
# 6. Final Dataset Check
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("6. Final Dataset Check")
print("=" * 50)

print(f"Total rows    : {len(df):,}")
print(f"Total columns : {df.shape[1]}")

print("\nRemaining missing values:")

missing = df.isnull().sum()
remaining = missing[missing > 0]

if len(remaining) == 0:
    print("None ✅")
else:
    print(remaining)

# ─────────────────────────────────────────
# 7. Save Processed Dataset
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("7. Saving Processed Dataset")
print("=" * 50)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset saved → {OUTPUT_PATH}")
