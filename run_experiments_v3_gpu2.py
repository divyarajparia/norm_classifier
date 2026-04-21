"""
Run Exp 1 + Exp 2 (v4, flan-t5 only) on GPU 2.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=2 python run_experiments_v3_gpu2.py
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
        "name": "Exp 1 v3: CultureBank Only",
        "train_csv": "train_culturebank.csv",
        "output_dir": "results_v4_exp1_culturebank",
    },
    {
        "name": "Exp 2 v3: Merged (CultureBank + NormAd)",
        "train_csv": "train_merged.csv",
        "output_dir": "results_v4_exp2_merged",
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
print("GPU 2 V3 EXPERIMENTS COMPLETE (Exp 1 + 2)")
print("="*80)
