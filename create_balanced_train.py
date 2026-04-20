"""
Create oversampled training set using simple random oversampling.
Duplicates existing rows for minority cultures until they reach the median count.
No API calls — just resampling existing data.

Input:  train_merged.csv
Output: train_merged_oversampled.csv
"""

import pandas as pd

INPUT_FILE = "train_merged.csv"
OUTPUT_FILE = "train_merged_oversampled.csv"

print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
df = df.dropna(subset=["culture", "norm", "generic"]).reset_index(drop=True)
print(f"Loaded {len(df)} pairs, {df['culture'].nunique()} cultures")

counts = df["culture"].value_counts()
target = int(counts.quantile(0.75))  # 75th percentile — aggressive oversampling
print(f"\nCulture count stats: min={counts.min()}, max={counts.max()}, median={counts.median():.0f}")
print(f"Oversampling target (75th percentile): {target}")

needs_oversampling = counts[counts < target]
print(f"Cultures below median: {len(needs_oversampling)}")

if len(needs_oversampling) == 0:
    print("No oversampling needed!")
    df.to_csv(OUTPUT_FILE, index=False)
else:
    extra_rows = []
    for culture, count in needs_oversampling.items():
        needed = target - count
        culture_df = df[df["culture"] == culture]
        # Sample with replacement to fill up to target
        sampled = culture_df.sample(n=needed, replace=True, random_state=42)
        extra_rows.append(sampled)
        print(f"  {culture}: {count} → {target} (+{needed} duplicated)")

    df_extra = pd.concat(extra_rows, ignore_index=True)
    df_combined = pd.concat([df, df_extra], ignore_index=True).reset_index(drop=True)

    df_combined.to_csv(OUTPUT_FILE, index=False)

    new_counts = df_combined["culture"].value_counts()
    print(f"\nOriginal: {len(df)} pairs")
    print(f"After oversampling: {len(df_combined)} pairs (+{len(df_extra)} duplicated)")
    print(f"New min: {new_counts.min()} | New median: {new_counts.median():.0f} | Max: {new_counts.max()}")
    print(f"Saved to: {OUTPUT_FILE}")
