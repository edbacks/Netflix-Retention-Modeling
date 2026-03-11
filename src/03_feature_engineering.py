"""
03_feature_engineering.py
Netflix Retention Modeling - Feature Engineering

This script converts session-level data into user-level features
for churn prediction modeling.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

INPUT_PATH = DATA_DIR / "watch_preprocessed.csv"
OUTPUT_PATH = DATA_DIR / "user_features.csv"

# ─────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────

def load_data(input_path: Path) -> pd.DataFrame:
    """Load preprocessed session-level data."""
    df = pd.read_csv(input_path, parse_dates=["watch_date", "last_watch_date"])
    return df


def aggregate_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session-level records into user-level features."""
    user_features = (
        df.groupby("user_id")
        .agg(
            total_sessions=("watch_date", "count"),
            total_watch_time=("watch_duration_minutes", "sum"),
            avg_watch_time=("watch_duration_minutes", "mean"),
            avg_completion_rate=("completion_rate", "mean"),
            last_watch_date=("watch_date", "max"),
            first_watch_date=("watch_date", "min"),
            genre_diversity=("genre_primary", "nunique"),
            device_diversity=("device_type", "nunique"),
            n_completed=("action", lambda x: (x == "completed").sum()),
            n_movies=("content_type", lambda x: (x == "Movie").sum()),
            n_original=("is_netflix_original", "sum"),
            churned=("churned", "max"),
        )
        .reset_index()
    )

    return user_features


def create_derived_features(user_features: pd.DataFrame) -> pd.DataFrame:
    """Create additional user-level behavioral features."""
    user_features["active_days"] = (
        user_features["last_watch_date"] - user_features["first_watch_date"]
    ).dt.days + 1

    user_features["active_days"] = user_features["active_days"].clip(lower=1)

    user_features["session_frequency"] = (
        user_features["total_sessions"] / user_features["active_days"]
    )

    user_features["completion_ratio"] = (
        user_features["n_completed"] / user_features["total_sessions"]
    )

    user_features["movie_ratio"] = (
        user_features["n_movies"] / user_features["total_sessions"]
    )

    user_features["original_ratio"] = (
        user_features["n_original"] / user_features["total_sessions"]
    )

    return user_features


def drop_intermediate_columns(user_features: pd.DataFrame) -> pd.DataFrame:
    """Drop intermediate columns not intended for final modeling."""
    drop_cols = [
        "last_watch_date",
        "first_watch_date",
        "n_completed",
        "n_movies",
        "n_original",
    ]

    existing_drop_cols = [col for col in drop_cols if col in user_features.columns]
    user_features = user_features.drop(columns=existing_drop_cols)

    return user_features


def print_feature_summary(user_features: pd.DataFrame, feature_cols: list[str]) -> None:
    """Print final feature list and descriptive statistics."""
    print("\n" + "=" * 50)
    print("Final Feature List")
    print("=" * 50)

    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")

    print(f"\nTotal users    : {len(user_features):,}")
    print(f"Total features : {len(feature_cols)}")

    print(
        f"Churned users  : {user_features['churned'].sum():,} "
        f"({user_features['churned'].mean() * 100:.1f}%)"
    )

    print("\n" + "=" * 50)
    print("Feature Summary Statistics")
    print("=" * 50)
    print(user_features[feature_cols].describe().round(2).to_string())


def save_features(user_features: pd.DataFrame, output_path: Path) -> None:
    """Save the final user-level feature table."""
    user_features.to_csv(output_path, index=False)

    print("\n" + "=" * 50)
    print("Saving Features")
    print("=" * 50)
    print(f"Saved file: {output_path}")


# ─────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────

def main() -> None:
    print("=" * 50)
    print("1. Loading Preprocessed Session Data")
    print("=" * 50)

    df = load_data(INPUT_PATH)

    print(f"Total rows    : {len(df):,} (session-level)")
    print(f"Unique users  : {df['user_id'].nunique():,}")

    print("\n" + "=" * 50)
    print("2. Aggregating to User Level")
    print("=" * 50)
    user_features = aggregate_user_features(df)

    print("\n" + "=" * 50)
    print("3. Creating Derived Features")
    print("=" * 50)
    user_features = create_derived_features(user_features)

    print("\n" + "=" * 50)
    print("4. Dropping Intermediate Columns")
    print("=" * 50)
    user_features = drop_intermediate_columns(user_features)

    feature_cols = [
        "total_sessions",
        "total_watch_time",
        "avg_watch_time",
        "avg_completion_rate",
        "active_days",
        "session_frequency",
        "genre_diversity",
        "device_diversity",
        "completion_ratio",
        "movie_ratio",
        "original_ratio",
    ]

    print_feature_summary(user_features, feature_cols)
    save_features(user_features, OUTPUT_PATH)


if __name__ == "__main__":
    main()
