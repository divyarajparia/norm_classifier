"""
Run Exp 3 + Exp 4 (v4, flan-t5 only) on GPU 3.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 python run_experiments_v3_gpu1.py
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
        "name": "Exp 3 v3: Merged + Oversampled",
        "train_csv": "train_merged_oversampled.csv",
        "output_dir": "results_v4_exp3_oversampled",
    },
    {
        "name": "Exp 4 v3: Merged + Synthetic",
        "train_csv": "train_merged_synthetic.csv",
        "output_dir": "results_v4_exp4_synthetic",
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
        "--stage2_epochs", "20",
        "--stage2_batch_size", "16",
        "--stage2_lr", "1e-5",
        "--gradient_accumulation_steps", "4",
        "--weighted_loss",
        "--models",
        "google/flan-t5-base",
    ]

    result = subprocess.run(cmd)

    elapsed = time.time() - start
    status = "DONE" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n>>> {exp['name']}: {status} in {elapsed/60:.1f} min")

print("\n" + "="*80)
print("GPU 1 V3 EXPERIMENTS COMPLETE (Exp 3 + 4)")
print("="*80)
