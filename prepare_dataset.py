"""
P3: Norm Classifier — Dataset Preparation
==========================================
Combines CultureBank (positives) and AG News (negatives)
into a balanced, labeled dataset split into train/val/test CSVs.
"""

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ── 1. LOAD CULTUREBANK (label = 1) ─────────────────────────────────────────
print("Loading CultureBank...")
cb = load_dataset("SALT-NLP/CultureBank")

df_tiktok = cb["tiktok"].to_pandas()
df_reddit  = cb["reddit"].to_pandas()

df_tiktok["source"] = "tiktok"
df_reddit["source"]  = "reddit"

df_cb = pd.concat([df_tiktok, df_reddit], ignore_index=True)
print(f"CultureBank total rows: {len(df_cb)}")

# Keep only high-agreement norms and drop missing actor_behavior
df_cb = df_cb[df_cb["agreement"] >= 0.6]
df_cb = df_cb[df_cb["actor_behavior"].notna()]
df_cb = df_cb[df_cb["actor_behavior"].str.strip() != ""]

print(f"CultureBank after filtering: {len(df_cb)}")

# Build the positive examples
df_pos = pd.DataFrame()
df_pos["text"]    = df_cb["actor_behavior"].str.strip()
df_pos["label"]   = 1
df_pos["culture"] = df_cb["cultural group"].values

# ── 2. LOAD AG NEWS (label = 0) ──────────────────────────────────────────────
print("\nLoading AG News...")
ag = load_dataset("ag_news", split="train")
df_ag = ag.to_pandas()
print(f"AG News total rows: {len(df_ag)}")

# Sample the same number as positives to keep classes balanced
n = len(df_pos)
df_ag = df_ag.sample(n=n, random_state=42).reset_index(drop=True)

# Build the negative examples
df_neg = pd.DataFrame()
df_neg["text"]    = df_ag["text"].str.strip()
df_neg["label"]   = 0
df_neg["culture"] = "generic"

print(f"AG News sampled: {len(df_neg)}")

# ── 3. COMBINE & SHUFFLE ─────────────────────────────────────────────────────
df_all = pd.concat([df_pos, df_neg], ignore_index=True)
df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset: {len(df_all)} rows")
print(f"Label distribution:\n{df_all['label'].value_counts()}")

# ── 4. TRAIN / VAL / TEST SPLIT (70 / 15 / 15) ──────────────────────────────
df_train, df_temp = train_test_split(
    df_all, test_size=0.30, random_state=42, stratify=df_all["label"]
)
df_val, df_test = train_test_split(
    df_temp, test_size=0.50, random_state=42, stratify=df_temp["label"]
)

print(f"\nSplit sizes:")
print(f"  Train : {len(df_train)}")
print(f"  Val   : {len(df_val)}")
print(f"  Test  : {len(df_test)}")

# ── 5. SAVE ──────────────────────────────────────────────────────────────────
df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv",     index=False)
df_test.to_csv("test.csv",   index=False)

print("\n✅ Saved: train.csv, val.csv, test.csv")
print("Columns: text | label | culture")
print("  text    — the sentence (norm or generic)")
print("  label   — 1 = norm, 0 = generic")
print("  culture — cultural group or 'generic'")
