"""
Run Exp 4 (v4, flan-t5 only) on GPU 3.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=3 python run_v4_exp4.py
"""
import os, subprocess, sys, time
os.environ["PYTHONNOUSERSITE"] = "1"

cmd = [
    sys.executable, "two_level_pipeline_v2.py",
    "--train_csv", "train_merged_synthetic.csv",
    "--test_csv", "test_culturebank.csv",
    "--output_dir", "results_v4_exp4_synthetic",
    "--epochs", "3", "--batch_size", "8",
    "--stage2_epochs", "20", "--stage2_batch_size", "16",
    "--stage2_lr", "1e-5", "--gradient_accumulation_steps", "4",
    "--weighted_loss",
    "--skip_stage1",
    "--models", "google/flan-t5-base",
]

start = time.time()
result = subprocess.run(cmd)
print(f"\n>>> Exp 4: {'DONE' if result.returncode == 0 else 'FAILED'} in {(time.time()-start)/60:.1f} min")
