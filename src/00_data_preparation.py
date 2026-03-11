"""
00_data_preparation.py
Netflix Retention Modeling - Data Preparation

This script:
1. Loads watch history and movie metadata
2. Removes duplicate movie records
3. Joins watch data with movie information
4. Creates derived behavioral features
5. Saves the processed dataset
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

WATCH_PATH = DATA_DIR / "watch_history.csv"
MOVIES_PATH = DATA_DIR / "movies.csv"
OUTPUT_PATH = DATA_DIR / "watch_joined.csv"

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────

print("=" * 50)
print("1. Loading datasets")
print("=" * 50)

watch = pd.read_csv(WATCH_PATH, parse_dates=["watch_date"])
movies = pd.read_csv(MOVIES_PATH)

print(f"watch_history : {len(watch):,} rows / {watch.shape[1]} columns")
print(f"movies        : {len(movies):,} rows / {movies.shape[1]} columns")

# ─────────────────────────────────────────
# 2. Remove duplicate movies
# Keep the first occurrence of each movie_id
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("2. Removing duplicate movies")
print("=" * 50)

before = len(movies)

movies = movies.drop_duplicates(
    subset=["movie_id"],
    keep="first"
)

after = len(movies)

print(f"Rows before duplicate removal : {before:,}")
print(f"Rows after duplicate removal  : {after:,}")
print(f"Duplicates removed            : {before - after:,}")

# ─────────────────────────────────────────
# 3. Join watch history with movie metadata
# LEFT JOIN on movie_id
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("3. Joining watch history with movie metadata")
print("=" * 50)

# Columns that are not needed for modeling
columns_to_drop = [
    "production_budget",
    "box_office_revenue",
    "number_of_seasons",
    "number_of_episodes",
    "added_to_platform",
    "content_warning"
]

# Only drop columns that actually exist
existing_cols = [c for c in columns_to_drop if c in movies.columns]
movies = movies.drop(columns=existing_cols)

df = watch.merge(
    movies,
    on="movie_id",
    how="left"
)

print(f"Joined dataset : {len(df):,} rows / {df.shape[1]} columns")

# ─────────────────────────────────────────
# 4. Create Derived Features
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("4. Creating derived features")
print("=" * 50)

# Completion rate
# percentage of content watched
df["completion_rate"] = df["progress_percentage"] / 100

print("completion_rate created")

# Watch ratio
# watch_duration / total content duration
df["watch_ratio"] = df["watch_duration_minutes"] / df["duration_minutes"]

# Cap extreme values
df["watch_ratio"] = df["watch_ratio"].clip(upper=3.0)

print("watch_ratio created (capped at 3.0)")

# ─────────────────────────────────────────
# 5. Dataset Summary
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("5. Dataset summary")
print("=" * 50)

print(f"Total rows      : {len(df):,}")
print(f"Total columns   : {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())

print(f"\nDate range      : {df['watch_date'].min().date()} → {df['watch_date'].max().date()}")
print(f"Unique users    : {df['user_id'].nunique():,}")
print(f"Unique content  : {df['movie_id'].nunique():,}")

# Check missing values
print("\nTop columns with missing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# ─────────────────────────────────────────
# 6. Save Output
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("6. Saving processed dataset")
print("=" * 50)

df.to_csv(OUTPUT_PATH, index=False)

print(f"Dataset saved to: {OUTPUT_PATH}")
