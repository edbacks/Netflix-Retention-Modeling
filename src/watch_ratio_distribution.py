"""
watch_ratio_distribution.py
Watch Duration Ratio Distribution Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────
# Paths
# ─────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────
df = pd.read_csv(DATA_DIR / "watch_joined.csv")
total = len(df)

# ─────────────────────────────────────────
# 2. Calculate Ratio
# ratio = watch_duration_minutes / duration_minutes
# ─────────────────────────────────────────
df_valid = df.dropna(subset=["watch_duration_minutes", "duration_minutes"]).copy()
df_valid["ratio"] = df_valid["watch_duration_minutes"] / df_valid["duration_minutes"]

missing  = total - len(df_valid)
under1   = (df_valid["ratio"] < 1).sum()
one_to3  = ((df_valid["ratio"] >= 1) & (df_valid["ratio"] < 3)).sum()
over3    = (df_valid["ratio"] >= 3).sum()

# ─────────────────────────────────────────
# 3. Print Results
# ─────────────────────────────────────────
print("=" * 50)
print("Watch Duration Ratio Distribution")
print("=" * 50)
print(f"Total sessions       : {total:,} (100%)")
print(f"ratio < 1  (normal)  : {under1:,} ({under1/total*100:.1f}%)")
print(f"ratio 1-3  (kept)    : {one_to3:,} ({one_to3/total*100:.1f}%)")
print(f"ratio >= 3 (removed) : {over3:,} ({over3/total*100:.1f}%)")
print(f"missing              : {missing:,} ({missing/total*100:.1f}%)")

# ─────────────────────────────────────────
# 4. Bar Chart
# ─────────────────────────────────────────
BG   = "#F8F8F8"
RED  = "#E50914"
DARK = "#221F1F"
GRAY = "#AAAAAA"

labels = [
    "ratio < 1\n(normal)",
    "ratio 1-3\n(kept)",
    "ratio >= 3\n(removed)",
    "missing",
]
values = [
    under1  / total * 100,
    one_to3 / total * 100,
    over3   / total * 100,
    missing / total * 100,
]
colors = [GRAY, "#FF6B6B", RED, "#CCCCCC"]
counts = [under1, one_to3, over3, missing]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white")

for bar, val, cnt in zip(bars, values, counts):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%\n({cnt:,})",
        ha="center", va="bottom",
        fontsize=9, color=DARK,
    )

ax.set_ylabel("Percentage (%)", color=DARK)
ax.set_title(
    "Watch Duration Ratio Distribution\n(watch_duration_minutes / duration_minutes)",
    fontsize=12, fontweight="bold", color=DARK, pad=15
)
ax.set_ylim(0, max(values) * 1.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(colors=DARK)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "watch_ratio_distribution.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()

print("-> outputs/watch_ratio_distribution.png saved")
