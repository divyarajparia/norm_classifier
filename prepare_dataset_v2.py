"""
Norm Classifier — Dataset Preparation v2
=========================================
Reads Gemini-generated pairs (norm + generic) from generated_pairs.csv
and assembles balanced train/val/test CSVs.

Requires: generated_pairs.csv (produced by generate_factual_negatives.py)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "generated_pairs_openai.csv"

# ── 1. LOAD GENERATED PAIRS ────────────────────────────────────────────────
print("Loading generated pairs...")
df_pairs = pd.read_csv(INPUT_FILE)
print(f"Total pairs: {len(df_pairs)}")
print(f"Cultures: {df_pairs['culture'].nunique()}")

# ── 2. BUILD NORM (label=1) AND GENERIC (label=0) ROWS ─────────────────────
df_norms = pd.DataFrame({
    "text": df_pairs["norm"],
    "label": 1,
    "culture": df_pairs["culture"],
})

df_generic = pd.DataFrame({
    "text": df_pairs["generic"],
    "label": 0,
    "culture": df_pairs["culture"],
})

# ── 3. COMBINE & SHUFFLE ───────────────────────────────────────────────────
df_all = pd.concat([df_norms, df_generic], ignore_index=True)

# Drop any rows with missing text
before = len(df_all)
df_all = df_all.dropna(subset=["text"])
df_all = df_all[df_all["text"].str.strip() != ""]
if before - len(df_all) > 0:
    print(f"Dropped {before - len(df_all)} empty rows")

# Drop duplicate texts
before = len(df_all)
df_all = df_all.drop_duplicates(subset=["text"])
if before - len(df_all) > 0:
    print(f"Dropped {before - len(df_all)} duplicate texts")

df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nCombined dataset: {len(df_all)} rows")
print(f"Label distribution:\n{df_all['label'].value_counts()}")

# ── 4. TRAIN / VAL / TEST SPLIT (70 / 15 / 15) ────────────────────────────
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

# ── 5. SAVE ─────────────────────────────────────────────────────────────────
df_train.to_csv("train.csv", index=False)
df_val.to_csv("val.csv", index=False)
df_test.to_csv("test.csv", index=False)

print("\nSaved: train.csv, val.csv, test.csv")
print("Columns: text | label | culture")

# ── 6. DIAGNOSTICS ──────────────────────────────────────────────────────────
print("\n=== DIAGNOSTICS ===")
for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
    norms = df[df["label"] == 1]
    generic = df[df["label"] == 0]
    norm_len = norms["text"].str.split().str.len().mean()
    gen_len = generic["text"].str.split().str.len().mean()
    print(f"  {name:5s}: {len(df):5d} rows | norms={len(norms)} generic={len(generic)} | "
          f"norm_avg_words={norm_len:.1f} generic_avg_words={gen_len:.1f}")

# Show sample pairs
print("\n=== SAMPLE PAIRS (from train) ===")
for culture in df_train["culture"].value_counts().head(5).index:
    norm_sample = df_train[(df_train["culture"] == culture) & (df_train["label"] == 1)].iloc[0]["text"]
    gen_sample = df_train[(df_train["culture"] == culture) & (df_train["label"] == 0)].iloc[0]["text"]
    print(f"\n  [{culture}]")
    print(f"    NORM:    {norm_sample[:100]}")
    print(f"    GENERIC: {gen_sample[:100]}")
