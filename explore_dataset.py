"""
P3: Norm Classifier — Dataset Exploration
==========================================
Loads CultureBank (TikTok + Reddit), merges them,
and produces a simple academic exploration of the data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# ── 1. LOAD & MERGE ──────────────────────────────────────────────────────────
print("Loading CultureBank...")
cb = load_dataset("SALT-NLP/CultureBank")

df_tiktok = cb["tiktok"].to_pandas()
df_reddit  = cb["reddit"].to_pandas()

df_tiktok["source"] = "tiktok"
df_reddit["source"]  = "reddit"

df_all = pd.concat([df_tiktok, df_reddit], ignore_index=True)

print(f"TikTok rows : {len(df_tiktok)}")
print(f"Reddit rows : {len(df_reddit)}")
print(f"Total rows  : {len(df_all)}")

# ── 2. SCHEMA ────────────────────────────────────────────────────────────────
print("\n=== COLUMNS & DTYPES ===")
print(df_all.dtypes)

print("\n=== MISSING VALUES ===")
print(df_all.isnull().sum())

# ── 3. SAMPLE ROWS ───────────────────────────────────────────────────────────
print("\n=== SAMPLE TIKTOK ROWS ===")
print(df_tiktok.head(3).to_string(index=False))

print("\n=== SAMPLE REDDIT ROWS ===")
print(df_reddit.head(3).to_string(index=False))

# ── 4. AGREEMENT SCORE ───────────────────────────────────────────────────────
print("\n=== AGREEMENT SCORE (overall) ===")
print(df_all["agreement"].describe())

print("\n=== AGREEMENT SCORE by source ===")
print(df_all.groupby("source")["agreement"].describe())

# ── 5. TOPIC DISTRIBUTION ────────────────────────────────────────────────────
print("\n=== TOP 10 TOPICS ===")
print(df_all["topic"].value_counts().head(10))

# ── 6. CULTURAL GROUP DISTRIBUTION ──────────────────────────────────────────
print("\n=== TOP 15 CULTURAL GROUPS ===")
print(df_all["cultural group"].value_counts().head(15))

# ── 7. PLOTS ─────────────────────────────────────────────────────────────────

# Source distribution
fig, ax = plt.subplots(figsize=(5, 4))
df_all["source"].value_counts().plot(kind="bar", ax=ax, color=["steelblue", "coral"])
ax.set_title("Rows per Source")
ax.set_xlabel("Source")
ax.set_ylabel("Count")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("plot_source_dist.png", dpi=150)
print("Saved: plot_source_dist.png")

# Top 10 topics
fig, ax = plt.subplots(figsize=(9, 5))
df_all["topic"].value_counts().head(10).sort_values().plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Top 10 Topics in CultureBank")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig("plot_topics.png", dpi=150)
print("Saved: plot_topics.png")

# Top 15 cultural groups
fig, ax = plt.subplots(figsize=(9, 5))
df_all["cultural group"].value_counts().head(15).sort_values().plot(kind="barh", ax=ax, color="coral")
ax.set_title("Top 15 Cultural Groups in CultureBank")
ax.set_xlabel("Count")
plt.tight_layout()
plt.savefig("plot_cultures.png", dpi=150)
print("Saved: plot_cultures.png")

# Agreement score distribution
fig, ax = plt.subplots(figsize=(7, 4))
df_all["agreement"].plot(kind="hist", bins=20, ax=ax, color="mediumseagreen", edgecolor="white")
ax.axvline(x=0.6, color="red", linestyle="--", label="Threshold (0.6)")
ax.set_title("Agreement Score Distribution")
ax.set_xlabel("Agreement Score")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig("plot_agreement.png", dpi=150)
print("Saved: plot_agreement.png")

print("\n✅ Exploration complete!")