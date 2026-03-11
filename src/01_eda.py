"""
01_eda.py
Netflix Retention Modeling - Exploratory Data Analysis

This script performs:
- Dataset overview
- Missing value analysis
- Outlier inspection
- Feature distribution analysis
- Churn label preview
- Visualization generation
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────
# Color Theme (Netflix style)
# ─────────────────────────────────────────

BG   = "#F8F8F8"
RED  = "#E50914"
DARK = "#221F1F"
GRAY = "#AAAAAA"

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────

df = pd.read_csv(DATA_DIR / "watch_joined.csv", parse_dates=["watch_date"])

print("=" * 50)
print("1. Dataset Overview")
print("=" * 50)

print(f"Total rows       : {len(df):,}")
print(f"Total columns    : {df.shape[1]}")
print(f"Unique users     : {df['user_id'].nunique():,}")
print(f"Unique content   : {df['movie_id'].nunique():,}")
print(f"Date range       : {df['watch_date'].min().date()} → {df['watch_date'].max().date()}")

# ─────────────────────────────────────────
# 2. Missing Value Analysis
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("2. Missing Value Analysis")
print("=" * 50)

missing = df.isnull().sum()
missing_pct = missing / len(df) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing,
    "Missing Rate (%)": missing_pct.round(1)
})

missing_df = missing_table[missing_table["Missing Count"] > 0]

print(missing_df)

# ─────────────────────────────────────────
# 3. Redundant Column Check
# completion_rate = progress_percentage / 100
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("3. Redundant Feature Check")
print("=" * 50)

both = df.dropna(subset=["completion_rate", "progress_percentage"])

match_pct = (
    both["completion_rate"].round(3) ==
    (both["progress_percentage"] / 100).round(3)
).mean() * 100

print(f"completion_rate == progress_percentage / 100 : {match_pct:.1f}% match")
print("Decision: drop progress_percentage")

# ─────────────────────────────────────────
# 4. Outlier Inspection
# ratio = watch_duration / duration
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("4. Outlier Inspection (watch_duration / duration_minutes)")
print("=" * 50)

df_valid = df.dropna(subset=["watch_duration_minutes", "duration_minutes"]).copy()
df_valid["ratio"] = df_valid["watch_duration_minutes"] / df_valid["duration_minutes"]

total = len(df)
missing_ratio = total - len(df_valid)

under1 = (df_valid["ratio"] < 1).sum()
one_to3 = ((df_valid["ratio"] >= 1) & (df_valid["ratio"] < 3)).sum()
over3 = (df_valid["ratio"] >= 3).sum()

print(f"Total sessions           : {total:,} (100%)")
print(f"Missing ratio            : {missing_ratio:,} ({missing_ratio/total*100:.1f}%)")
print(f"ratio < 1  (normal)      : {under1:,} ({under1/total*100:.1f}%)")
print(f"ratio 1–3  (kept)        : {one_to3:,} ({one_to3/total*100:.1f}%)")
print(f"ratio ≥ 3  (removed)     : {over3:,} ({over3/total*100:.1f}%)")

before = set(df["user_id"].unique())
after = set(df_valid[df_valid["ratio"] < 3]["user_id"].unique())
lost_users = before - after

print(f"Users losing all sessions after filtering: {len(lost_users)}")

# ─────────────────────────────────────────
# 5. Numerical Feature Distribution
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("5. Numerical Feature Summary")
print("=" * 50)

print(
    df[
        [
            "watch_duration_minutes",
            "completion_rate",
            "imdb_rating"
        ]
    ].describe().round(2)
)

# ─────────────────────────────────────────
# 6. Categorical Feature Distribution
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("6. Categorical Feature Distribution")
print("=" * 50)

for col in ["device_type", "action", "genre_primary", "content_type"]:
    print(f"\n[{col}]")
    print(df[col].value_counts().to_string())

# ─────────────────────────────────────────
# 7. Sessions per User
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("7. Sessions per User Distribution")
print("=" * 50)

user_sessions = df.groupby("user_id").size()

print(user_sessions.describe().round(1))

# ─────────────────────────────────────────
# 8. Churn Label Preview
# 30-day inactivity rule
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("8. Churn Label Preview (30-day inactivity rule)")
print("=" * 50)

reference_date = df["watch_date"].max()
last_watch = df.groupby("user_id")["watch_date"].max()

recency_days = (reference_date - last_watch).dt.days
churned = (recency_days >= 30).astype(int)

print(f"Churned users   : {churned.sum():,} ({churned.mean()*100:.1f}%)")
print(f"Retained users  : {(1 - churned).sum():,} ({(1 - churned).mean()*100:.1f}%)")

# ─────────────────────────────────────────
# 9. Feature Removal Summary
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("9. Feature Removal Decisions")
print("=" * 50)

drop_reasons = {
    "progress_percentage": "Duplicate of completion_rate",
    "watch_ratio": "Derived feature (recomputable)",
    "user_rating": "79.9% missing (selection bias)",
    "genre_secondary": "64.4% missing values",
    "session_id": "Identifier only (not useful for modeling)"
}

for col, reason in drop_reasons.items():
    print(f"{col:<25} → {reason}")

# ─────────────────────────────────────────
# 10. Visualization Export
# ─────────────────────────────────────────

print("\n" + "=" * 50)
print("10. Generating EDA Visualizations")
print("=" * 50)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(BG)

axes = axes.flatten()

def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, fontsize=10, fontweight="bold", color=DARK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ① Missing value ratio

ax = axes[0]

cols = missing_df.index.tolist()
vals = missing_df["Missing Rate (%)"].tolist()

colors = [RED if v > 50 else "#FF6B6B" if v > 10 else GRAY for v in vals]

ax.barh(cols, vals, color=colors)

ax.set_xlabel("Missing Rate (%)")

style(ax, "Missing Value Ratio")

# ② Ratio distribution

ax = axes[1]

ratio_labels = [
    "ratio < 1\n(normal)",
    "ratio 1–3\n(kept)",
    "ratio ≥ 3\n(outlier)",
    "missing"
]

ratio_vals = [
    under1 / total * 100,
    one_to3 / total * 100,
    over3 / total * 100,
    missing_ratio / total * 100
]

ax.bar(ratio_labels, ratio_vals, color=[GRAY, "#FF6B6B", RED, "#CCCCCC"])

ax.set_ylabel("Percentage (%)")

style(ax, "Watch Ratio Distribution")

# ③ Churn vs retained

ax = axes[2]

ax.pie(
    [churned.sum(), (1 - churned).sum()],
    labels=["Churned", "Retained"],
    colors=[RED, DARK],
    autopct="%1.1f%%"
)

style(ax, "Churn vs Retained")

# ④ Watch duration distribution

ax = axes[3]

dur = df["watch_duration_minutes"].dropna().clip(0, 300)

ax.hist(dur, bins=40, color=RED)

ax.set_xlabel("Watch Duration (minutes)")
ax.set_ylabel("Frequency")

style(ax, "Session Watch Duration")

# ⑤ Genre distribution

ax = axes[4]

genre = df["genre_primary"].value_counts().head(8)

ax.barh(genre.index[::-1], genre.values[::-1], color=GRAY)

ax.set_xlabel("Session Count")

style(ax, "Top Genres")

# ⑥ Completion rate distribution

ax = axes[5]

comp = df["completion_rate"].dropna()

ax.hist(comp, bins=30, color=DARK)

ax.set_xlabel("Completion Rate")

style(ax, "Completion Rate Distribution")

fig.suptitle(
    "Netflix Retention Modeling — EDA Overview",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_eda_overview.png",
    dpi=150,
    bbox_inches="tight",
    facecolor=BG
)

plt.close()

print("EDA visualization saved → outputs/01_eda_overview.png")
