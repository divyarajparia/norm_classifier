"""
Run Exp 3 + Exp 4 on GPU 3.
Run this in one terminal: CUDA_VISIBLE_DEVICES=1 python run_experiments_gpu3.py
"""

import os
import subprocess
import sys
import time

os.environ["PYTHONNOUSERSITE"] = "1"
PYTHON = sys.executable
PIPELINE = "two_level_pipeline_v2.py"
TEST_CSV = "test_culturebank.csv"

experiments = [
    {
        "name": "Exp 3: Merged + Oversampled",
        "train_csv": "train_merged_oversampled.csv",
        "output_dir": "results_exp3_oversampled",
    },
    {
        "name": "Exp 4: Merged + Synthetic",
        "train_csv": "train_merged_synthetic.csv",
        "output_dir": "results_exp4_synthetic",
    },
]

for exp in experiments:
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
        "--epochs", "3",
        "--batch_size", "8",
        "--stage2_epochs", "8",
        "--stage2_batch_size", "64",
        "--models",
        "roberta-base",
        "google/flan-t5-base",
    ]

    result = subprocess.run(cmd)

    elapsed = time.time() - start
    status = "DONE" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n>>> {exp['name']}: {status} in {elapsed/60:.1f} min")

print("\n" + "="*80)
print("GPU 3 EXPERIMENTS COMPLETE (Exp 3 + 4)")
print("="*80)
