"""
Run all 4 experiments sequentially using two_level_pipeline_v2.py.
Each experiment uses a different train CSV but the SAME test CSV (test_culturebank.csv).
"""

import subprocess
import sys
import time

PYTHON = sys.executable
PIPELINE = "two_level_pipeline_v2.py"
TEST_CSV = "test_culturebank.csv"

experiments = [
    # Exp 1: CultureBank only (ALREADY DONE — uncomment to rerun)
    # {
    #     "name": "Exp 1: CultureBank Only",
    #     "train_csv": "train_culturebank.csv",
    #     "output_dir": "results_exp1_culturebank",
    # },

    # Exp 2: CultureBank + NormAd merged
    {
        "name": "Exp 2: Merged (CultureBank + NormAd)",
        "train_csv": "train_merged.csv",
        "output_dir": "results_exp2_merged",
    },

    # Exp 3: Merged + simple oversampling (duplicate minority rows)
    {
        "name": "Exp 3: Merged + Oversampled",
        "train_csv": "train_merged_oversampled.csv",
        "output_dir": "results_exp3_oversampled",
    },

    # Exp 4: Merged + LLM-generated synthetic norms for minorities
    {
        "name": "Exp 4: Merged + Synthetic",
        "train_csv": "train_merged_synthetic.csv",
        "output_dir": "results_exp4_synthetic",
    },
]

for i, exp in enumerate(experiments):
    print(f"\n{'='*80}")
    print(f"  {exp['name']}")
    print(f"  Train: {exp['train_csv']} | Test: {TEST_CSV}")
    print(f"  Output: {exp['output_dir']}")
    print(f"{'='*80}\n")

    start = time.time()

    cmd = [
        PYTHON, PIPELINE,
        "--train_csv", exp["train_csv"],
        "--test_csv", TEST_CSV,
        "--output_dir", exp["output_dir"],
        "--epochs", "1",
        "--batch_size", "8",
    ]

    result = subprocess.run(cmd)

    elapsed = time.time() - start
    status = "DONE" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n>>> {exp['name']}: {status} in {elapsed/60:.1f} min")

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE")
print("="*80)
