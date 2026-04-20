"""
Create fixed 80/20 train/test split of CultureBank pairs.
Split at the pairs level (norm+generic stay together).
Stratified on culture. seed=42 to match two_level_pipeline.py.

Output:
  train_culturebank.csv  (~80% of pairs)
  test_culturebank.csv   (~20% of pairs)

Columns: culture, original_norm, norm, generic
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "generated_pairs_openai.csv"
TRAIN_CSV = "train_culturebank.csv"
TEST_CSV = "test_culturebank.csv"
TEST_SIZE = 0.2
SEED = 42

df = pd.read_csv(INPUT_CSV)
df = df.dropna(subset=["culture", "norm", "generic"]).reset_index(drop=True)

print(f"Loaded {len(df)} pairs from {INPUT_CSV}")
print(f"Cultures: {df['culture'].nunique()}")

# Drop cultures with <2 samples (can't stratify with 1 sample)
counts = df["culture"].value_counts()
rare = counts[counts < 2].index
if len(rare) > 0:
    print(f"Dropping {len(rare)} cultures with <2 samples: {list(rare)}")
    df = df[~df["culture"].isin(rare)].reset_index(drop=True)

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["culture"],
)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df.to_csv(TRAIN_CSV, index=False)
test_df.to_csv(TEST_CSV, index=False)

print(f"\nTrain: {len(train_df)} pairs → {TRAIN_CSV}")
print(f"Test:  {len(test_df)} pairs → {TEST_CSV}")

print(f"\nTrain cultures: {train_df['culture'].nunique()}")
print(f"Test cultures:  {test_df['culture'].nunique()}")

print("\nTrain top 5 cultures:")
print(train_df["culture"].value_counts().head())
print("\nTest top 5 cultures:")
print(test_df["culture"].value_counts().head())
