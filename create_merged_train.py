"""
Create merged training set: CultureBank train + full NormAd pairs.
Test set remains test_culturebank.csv (unchanged).

Input:
  train_culturebank.csv     (from create_culturebank_split.py)
  normad_pairs_openai.csv   (from generate_normad_pairs.py)

Output:
  train_merged.csv          (CultureBank train + all NormAd)
"""

import pandas as pd

TRAIN_CB = "train_culturebank.csv"
NORMAD_PAIRS = "normad_pairs_openai.csv"
OUTPUT_CSV = "train_merged.csv"

print("Loading datasets...")
df_cb = pd.read_csv(TRAIN_CB)
df_normad = pd.read_csv(NORMAD_PAIRS)

print(f"CultureBank train: {len(df_cb)} pairs, {df_cb['culture'].nunique()} cultures")
print(f"NormAd pairs:      {len(df_normad)} pairs, {df_normad['culture'].nunique()} cultures")

# Ensure same columns
cols = ["culture", "original_norm", "norm", "generic"]
df_cb = df_cb[cols]
df_normad = df_normad[cols]

# Concatenate
df_merged = pd.concat([df_cb, df_normad], ignore_index=True)

# Drop exact duplicate norms
before = len(df_merged)
df_merged = df_merged.drop_duplicates(subset=["norm"]).reset_index(drop=True)
dupes = before - len(df_merged)
if dupes > 0:
    print(f"Dropped {dupes} duplicate norms")

df_merged.to_csv(OUTPUT_CSV, index=False)

print(f"\nMerged train: {len(df_merged)} pairs, {df_merged['culture'].nunique()} cultures")
print(f"Saved to: {OUTPUT_CSV}")

# Show culture distribution summary
vc = df_merged["culture"].value_counts()
print(f"\nTop 10 cultures:")
print(vc.head(10).to_string())
print(f"\nBottom 10 cultures:")
print(vc.tail(10).to_string())
print(f"\nMin: {vc.min()} ({vc.idxmin()}) | Max: {vc.max()} ({vc.idxmax()}) | Median: {vc.median():.0f}")
