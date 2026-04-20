"""
Two-level norm classification pipeline (v2).
Same as two_level_pipeline.py but reads pre-split train/test CSVs
instead of creating splits internally.

Usage:
  # Exp 1: CultureBank only
  python two_level_pipeline_v2.py --train_csv train_culturebank.csv --test_csv test_culturebank.csv

  # Exp 2: Merged train
  python two_level_pipeline_v2.py --train_csv train_merged.csv --test_csv test_culturebank.csv

  # Exp 3: Oversampled train
  python two_level_pipeline_v2.py --train_csv train_merged_oversampled.csv --test_csv test_culturebank.csv

Train/test CSVs must have columns: culture, original_norm, norm, generic
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    set_seed,
)


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SplitData:
    train_texts: List[str]
    train_labels: List[int]
    test_texts: List[str]
    test_labels: List[int]


def prepare_binary_dataset(train_pairs: pd.DataFrame, test_pairs: pd.DataFrame) -> SplitData:
    """Build binary (norm=1, generic=0) dataset from pre-split pairs."""
    def pairs_to_binary(df):
        df_norm = pd.DataFrame({"text": df["norm"].astype(str).str.strip(), "label": 1})
        df_fact = pd.DataFrame({"text": df["generic"].astype(str).str.strip(), "label": 0})
        combined = pd.concat([df_norm, df_fact], ignore_index=True)
        combined = combined.dropna(subset=["text"])
        combined = combined[combined["text"] != ""]
        combined = combined.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
        return combined

    train_df = pairs_to_binary(train_pairs)
    test_df = pairs_to_binary(test_pairs)

    return SplitData(
        train_texts=train_df["text"].tolist(),
        train_labels=train_df["label"].tolist(),
        test_texts=test_df["text"].tolist(),
        test_labels=test_df["label"].tolist(),
    )


def prepare_normlabel_dataset(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
) -> Tuple[SplitData, Dict[str, int], Dict[int, str], pd.DataFrame]:
    """Build culture classification dataset from pre-split pairs (norm rows only)."""
    def pairs_to_norm_rows(df):
        result = pd.DataFrame({
            "text": df["norm"].astype(str).str.strip(),
            "culture": df["culture"].astype(str).str.strip(),
        })
        result = result.dropna(subset=["text", "culture"])
        result = result[(result["text"] != "") & (result["culture"] != "")]
        return result

    train_df = pairs_to_norm_rows(train_pairs)
    test_df = pairs_to_norm_rows(test_pairs)

    # Build label map from ALL cultures seen in train + test
    all_cultures = sorted(set(train_df["culture"].unique()) | set(test_df["culture"].unique()))
    label2id = {name: i for i, name in enumerate(all_cultures)}
    id2label = {i: name for name, i in label2id.items()}

    # Drop cultures with <2 samples in train (can't learn from 1 example)
    train_counts = train_df["culture"].value_counts()
    rare = train_counts[train_counts < 2].index
    if len(rare) > 0:
        print(f"Dropping {len(rare)} rare train labels (<2 samples): {list(rare)}")
        train_df = train_df[~train_df["culture"].isin(rare)].reset_index(drop=True)
        test_df = test_df[~test_df["culture"].isin(rare)].reset_index(drop=True)

        # Rebuild label map
        all_cultures = sorted(set(train_df["culture"].unique()) | set(test_df["culture"].unique()))
        label2id = {name: i for i, name in enumerate(all_cultures)}
        id2label = {i: name for name, i in label2id.items()}

    train_df["label"] = train_df["culture"].map(label2id)
    test_df["label"] = test_df["culture"].map(label2id)

    return (
        SplitData(
            train_texts=train_df["text"].tolist(),
            train_labels=train_df["label"].tolist(),
            test_texts=test_df["text"].tolist(),
            test_labels=test_df["label"].tolist(),
        ),
        label2id,
        id2label,
        test_df,
    )


def build_cls_dataset(texts: List[str], labels: List[int]) -> Dataset:
    return Dataset.from_dict({"text": texts, "label": labels})


def train_eval_encoder(
    model_name: str,
    train_data: SplitData,
    output_dir: str,
    num_labels: int,
    epochs: int,
    batch_size: int,
    max_length: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = build_cls_dataset(train_data.train_texts, train_data.train_labels)
    test_ds = build_cls_dataset(train_data.test_texts, train_data.test_labels)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    trainer.train()
    pred_out = trainer.predict(test_ds)
    preds = np.argmax(pred_out.predictions, axis=1)
    labels = np.array(train_data.test_labels)
    return preds, labels


def build_t5_dataset(texts: List[str], labels_text: List[str], prefix: str) -> Dataset:
    inputs = [f"{prefix} {t}" for t in texts]
    return Dataset.from_dict({"input_text": inputs, "target_text": labels_text})


def train_eval_t5(
    model_name: str,
    train_texts: List[str],
    train_label_texts: List[str],
    test_texts: List[str],
    test_label_texts: List[str],
    output_dir: str,
    epochs: int,
    batch_size: int,
    max_input_length: int,
    max_target_length: int,
    seed: int,
    prefix: str,
) -> Tuple[List[str], List[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_ds = build_t5_dataset(train_texts, train_label_texts, prefix)
    test_ds = build_t5_dataset(test_texts, test_label_texts, prefix)

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=max_input_length,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            truncation=True,
            max_length=max_target_length,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = train_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="no",
        save_strategy="no",
        predict_with_generate=True,
        generation_max_length=max_target_length,
        logging_steps=50,
        report_to=[],
        seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()
    pred_out = trainer.predict(test_ds)
    pred_text = tokenizer.batch_decode(pred_out.predictions, skip_special_tokens=True)
    pred_text = [p.strip() for p in pred_text]
    return pred_text, test_label_texts


def labelwise_accuracy(y_true: List[int], y_pred: List[int], id2label: Dict[int, str]) -> pd.DataFrame:
    rows = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    for label_id in sorted(np.unique(y_true_arr)):
        mask = y_true_arr == label_id
        count = int(mask.sum())
        acc = float((y_pred_arr[mask] == y_true_arr[mask]).mean()) if count > 0 else np.nan
        rows.append({"label_id": int(label_id), "label": id2label[int(label_id)], "count": count, "accuracy": acc})
    return pd.DataFrame(rows)


def slugify(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description="Two-level norm classification pipeline (v2 - pre-split)")
    parser.add_argument("--train_csv", required=True, help="Path to train pairs CSV (culture, original_norm, norm, generic)")
    parser.add_argument("--test_csv", required=True, help="Path to test pairs CSV (same columns)")
    parser.add_argument("--output_dir", default="two_level_results_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "bert-base-uncased",
            "distilbert-base-uncased",
            "roberta-base",
            "microsoft/deberta-v3-base",
            "google/flan-t5-base",
            "sentence-transformers/all-MiniLM-L6-v2",
            "albert-base-v2",
            "google/electra-base-discriminator",
            "xlm-roberta-base",
        ],
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Device available: {get_device()}")
    print(f"Loading train: {args.train_csv}")
    print(f"Loading test:  {args.test_csv}")

    train_pairs = pd.read_csv(args.train_csv)
    test_pairs = pd.read_csv(args.test_csv)

    required_cols = {"culture", "norm", "generic"}
    for name, df in [("train", train_pairs), ("test", test_pairs)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {missing}")

    train_pairs = train_pairs.dropna(subset=["culture", "norm", "generic"]).reset_index(drop=True)
    test_pairs = test_pairs.dropna(subset=["culture", "norm", "generic"]).reset_index(drop=True)

    print(f"Train pairs: {len(train_pairs)} ({train_pairs['culture'].nunique()} cultures)")
    print(f"Test pairs:  {len(test_pairs)} ({test_pairs['culture'].nunique()} cultures)")

    binary_data = prepare_binary_dataset(train_pairs, test_pairs)
    norm_data, label2id, id2label, norm_test_df = prepare_normlabel_dataset(train_pairs, test_pairs)

    # Persist splits
    pd.DataFrame({"text": binary_data.train_texts, "label": binary_data.train_labels}).to_csv(
        os.path.join(args.output_dir, "binary_train.csv"), index=False
    )
    pd.DataFrame({"text": binary_data.test_texts, "label": binary_data.test_labels}).to_csv(
        os.path.join(args.output_dir, "binary_test.csv"), index=False
    )
    pd.DataFrame({"text": norm_data.train_texts, "label": norm_data.train_labels}).to_csv(
        os.path.join(args.output_dir, "normlabel_train.csv"), index=False
    )
    pd.DataFrame({"text": norm_data.test_texts, "label": norm_data.test_labels, "culture": norm_test_df["culture"].tolist()}).to_csv(
        os.path.join(args.output_dir, "normlabel_test.csv"), index=False
    )
    with open(os.path.join(args.output_dir, "label_map.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

    print("Saved transformed datasets and label map.")
    print(f"Binary split: train={len(binary_data.train_texts)} test={len(binary_data.test_texts)}")
    print(f"Norm-label split: train={len(norm_data.train_texts)} test={len(norm_data.test_texts)} labels={len(label2id)}")

    summary_rows = []
    labelwise_all_rows = []

    for model_name in args.models:
        print("\n" + "=" * 90)
        print(f"MODEL: {model_name}")
        print("=" * 90)
        model_slug = slugify(model_name)

        binary_acc = np.nan
        norm_acc = np.nan
        stage1_status = "ok"
        stage2_status = "ok"

        try:
            if "flan-t5" in model_name.lower() or model_name.lower().startswith("t5"):
                train_lbl_txt = ["norm" if y == 1 else "fact" for y in binary_data.train_labels]
                test_lbl_txt = ["norm" if y == 1 else "fact" for y in binary_data.test_labels]
                pred_txt, gold_txt = train_eval_t5(
                    model_name=model_name,
                    train_texts=binary_data.train_texts,
                    train_label_texts=train_lbl_txt,
                    test_texts=binary_data.test_texts,
                    test_label_texts=test_lbl_txt,
                    output_dir=os.path.join(args.output_dir, f"tmp_{model_slug}_stage1"),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_input_length=args.max_length,
                    max_target_length=8,
                    seed=args.seed,
                    prefix="classify as norm or fact:",
                )
                pred_bin = [1 if normalize_label(p) == "norm" else 0 for p in pred_txt]
                gold_bin = [1 if g == "norm" else 0 for g in gold_txt]
                binary_acc = accuracy_score(gold_bin, pred_bin)
            else:
                pred, gold = train_eval_encoder(
                    model_name=model_name,
                    train_data=binary_data,
                    output_dir=os.path.join(args.output_dir, f"tmp_{model_slug}_stage1"),
                    num_labels=2,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    seed=args.seed,
                )
                binary_acc = accuracy_score(gold, pred)

            print(f"Stage 1 (Norm vs Fact) accuracy: {binary_acc:.4f}")
        except Exception as e:
            stage1_status = f"failed: {e}"
            print(f"Stage 1 failed for {model_name}: {e}")

        try:
            if "flan-t5" in model_name.lower() or model_name.lower().startswith("t5"):
                train_lbl_txt = [id2label[y] for y in norm_data.train_labels]
                test_lbl_txt = [id2label[y] for y in norm_data.test_labels]
                pred_txt, gold_txt = train_eval_t5(
                    model_name=model_name,
                    train_texts=norm_data.train_texts,
                    train_label_texts=train_lbl_txt,
                    test_texts=norm_data.test_texts,
                    test_label_texts=test_lbl_txt,
                    output_dir=os.path.join(args.output_dir, f"tmp_{model_slug}_stage2"),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_input_length=args.max_length,
                    max_target_length=16,
                    seed=args.seed,
                    prefix="classify norm culture label:",
                )
                norm_pred_ids = [label2id.get(p, -1) for p in pred_txt]
                norm_gold_ids = [label2id[g] for g in gold_txt]
                norm_acc = accuracy_score(norm_gold_ids, norm_pred_ids)
            else:
                pred, gold = train_eval_encoder(
                    model_name=model_name,
                    train_data=norm_data,
                    output_dir=os.path.join(args.output_dir, f"tmp_{model_slug}_stage2"),
                    num_labels=len(label2id),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    seed=args.seed,
                )
                norm_pred_ids = pred.tolist()
                norm_gold_ids = gold.tolist()
                norm_acc = accuracy_score(norm_gold_ids, norm_pred_ids)

            lw = labelwise_accuracy(norm_gold_ids, norm_pred_ids, id2label)
            lw["model"] = model_name
            labelwise_all_rows.append(lw)
            lw.to_csv(os.path.join(args.output_dir, f"labelwise_{model_slug}.csv"), index=False)

            print(f"Stage 2 (Norm label) accuracy: {norm_acc:.4f}")
        except Exception as e:
            stage2_status = f"failed: {e}"
            print(f"Stage 2 failed for {model_name}: {e}")

        summary_rows.append(
            {
                "model": model_name,
                "binary_test_accuracy": binary_acc,
                "norm_label_test_accuracy": norm_acc,
                "binary_train_size": len(binary_data.train_texts),
                "binary_test_size": len(binary_data.test_texts),
                "norm_train_size": len(norm_data.train_texts),
                "norm_test_size": len(norm_data.test_texts),
                "n_norm_labels": len(label2id),
                "stage1_status": stage1_status,
                "stage2_status": stage2_status,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "train_csv": args.train_csv,
                "test_csv": args.test_csv,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)

    if labelwise_all_rows:
        labelwise_df = pd.concat(labelwise_all_rows, ignore_index=True)
        labelwise_df.to_csv(os.path.join(args.output_dir, "summary_labelwise_all_models.csv"), index=False)

    print("\n=== Accuracy Summary (Test) ===")
    display_cols = ["model", "binary_test_accuracy", "norm_label_test_accuracy", "stage1_status", "stage2_status"]
    display_df = summary_df[display_cols].copy()
    print(display_df.to_string(index=False))

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
