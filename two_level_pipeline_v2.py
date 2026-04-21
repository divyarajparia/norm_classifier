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


# Demonym → country name mapping for culture stripping
DEMONYM_TO_COUNTRY = {
    "afghan": "afghanistan", "argentine": "argentina", "australian": "australia",
    "austrian": "austria", "bangladeshi": "bangladesh", "bosnian": "bosnia",
    "brazilian": "brazil", "cambodian": "cambodia", "canadian": "canada",
    "chilean": "chile", "chinese": "china", "colombian": "colombia",
    "croatian": "croatia", "cypriot": "cyprus", "egyptian": "egypt",
    "ethiopian": "ethiopia", "fijian": "fiji", "french": "france",
    "german": "germany", "greek": "greece", "hungarian": "hungary",
    "indian": "india", "indonesian": "indonesia", "iranian": "iran",
    "iraqi": "iraq", "irish": "ireland", "israeli": "israel",
    "italian": "italy", "japanese": "japan", "kenyan": "kenya",
    "laotian": "laos", "lebanese": "lebanon", "malaysian": "malaysia",
    "maltese": "malta", "mauritian": "mauritius", "mexican": "mexico",
    "burmese": "myanmar", "nepalese": "nepal", "dutch": "netherlands",
    "pakistani": "pakistan", "palestinian": "palestine", "peruvian": "peru",
    "filipino": "philippines", "polish": "poland", "portuguese": "portugal",
    "romanian": "romania", "russian": "russia", "samoan": "samoa",
    "serbian": "serbia", "singaporean": "singapore", "somali": "somalia",
    "spanish": "spain", "sudanese": "sudan", "swedish": "sweden",
    "syrian": "syria", "taiwanese": "taiwan", "thai": "thailand",
    "timorese": "timor-leste", "tongan": "tonga", "turkish": "turkey",
    "ukrainian": "ukraine", "british": "britain", "american": "america",
    "venezuelan": "venezuela", "vietnamese": "vietnam", "zimbabwean": "zimbabwe",
    "korean": "korea", "swiss": "switzerland", "belgian": "belgium",
    "norwegian": "norway", "danish": "denmark", "finnish": "finland",
    "czech": "czech republic", "albanian": "albania", "macedonian": "macedonia",
    "sri lankan": "sri lanka", "south african": "south africa",
    "south korean": "south korea", "south sudanese": "south sudan",
    "saudi arabian": "saudi arabia", "new zealander": "new zealand",
    "hong konger": "hong kong", "papua new guinean": "papua new guinea",
}

# ── Culture label cleanup ────────────────────────────────────────────────────
# Merge duplicate / overlapping culture labels to canonical form
CULTURE_MERGE_MAP = {
    # Plural → Singular
    "African Americans": "African American", "Arabs": "Arab",
    "Argentinians": "Argentinian", "Australians": "Australian",
    "Austrians": "Austrian", "Belgians": "Belgian",
    "Brazilians": "Brazilian", "Bulgarians": "Bulgarian",
    "Canadians": "Canadian", "Czechs": "Czech",
    "Egyptians": "Egyptian", "Filipinos": "Filipino",
    "Germans": "German", "Greeks": "Greek",
    "Indians": "Indian", "Indonesians": "Indonesian",
    "Israelis": "Israeli", "Italians": "Italian",
    "Latin Americans": "Latin American", "Malaysians": "Malaysian",
    "Mexicans": "Mexican", "New Zealanders": "New Zealander",
    "Nigerians": "Nigerian", "Norwegians": "Norwegian",
    "Puerto Ricans": "Puerto Rican", "Romanians": "Romanian",
    "Russians": "Russian", "Singaporeans": "Singaporean",
    "South Africans": "South African", "Ukrainians": "Ukrainian",
    "Chileans": "Chilean", "Colombians": "Colombian",
    "Venezuelans": "Venezuelan", "Peruvians": "Peruvian",
    "Panamanians": "Panamanian", "Moroccans": "Moroccan",
    "Pakistanis": "Pakistani", "Hungarians": "Hungarian",
    "Estonians": "Estonian", "Lithuanians": "Lithuanian",
    # Synonyms
    "Argentine": "Argentinian", "Finns": "Finnish",
    "Swedes": "Swedish", "Turks": "Turkish",
    "Spaniards": "Spanish", "Nederlanders": "Dutch",
    "Malay": "Malaysian", "Black Americans": "African American",
    # Compound → simple
    "Indian and Indian American": "Indian",
    "Italian and Italian-American": "Italian",
    "Mexican and Mexican American": "Mexican",
    "Latin America": "Latin American",
    # UK sub-regions → British
    "English": "British", "Londoners": "British",
    # Canada sub-regions → Canadian
    "Torontonians": "Canadian", "Québécois": "Canadian",
    # US sub-regions → American
    "Southerners": "American", "Southern Americans": "American",
    "Midwesterners": "American", "Minnesotans": "American",
    "Californians": "American", "Southern Californians": "American",
    "Floridians": "American", "Oregonians": "American",
    "Texans": "American", "New Yorkers": "American",
    "Washingtonians": "American", "Chicagoans": "American",
    "Charlotte residents": "American", "Hawaiians": "American",
    "North Americans": "American",
    # Sub-culture → parent
    "Sicilian": "Italian", "Malayalees": "Indian", "Balinese": "Indonesian",
}

# Meta-categories too broad to classify — drop from Stage 2
META_CULTURES_TO_DROP = {
    "Asian", "Asians", "European", "Europeans", "African",
    "Middle Eastern", "Nordic", "Scandinavians", "Westerners",
    "Hispanic", "Latinos",
}

# ── Region mapping ────────────────────────────────────────────────────────────
CULTURE_TO_REGION = {
    # East Asia
    "Chinese": "East Asia", "Japanese": "East Asia", "Korean": "East Asia",
    "South Korean": "East Asia", "North Korean": "East Asia", "Taiwanese": "East Asia",
    "Hong Konger": "East Asia",
    # Southeast Asia
    "Filipino": "Southeast Asia", "Indonesian": "Southeast Asia",
    "Malaysian": "Southeast Asia", "Singaporean": "Southeast Asia",
    "Thai": "Southeast Asia", "Vietnamese": "Southeast Asia",
    "Cambodian": "Southeast Asia", "Laotian": "Southeast Asia",
    "Burmese": "Southeast Asia", "Timorese": "Southeast Asia",
    # South Asia
    "Indian": "South Asia", "Pakistani": "South Asia",
    "Bangladeshi": "South Asia", "Nepalese": "South Asia",
    "Sri Lankan": "South Asia", "Afghan": "South Asia",
    # Middle East & North Africa
    "Arab": "MENA", "Egyptian": "MENA", "Iranian": "MENA",
    "Iraqi": "MENA", "Israeli": "MENA", "Lebanese": "MENA",
    "Palestinian": "MENA", "Saudi Arabian": "MENA", "Syrian": "MENA",
    "Turkish": "MENA", "Tunisian": "MENA", "Moroccan": "MENA",
    "Jordanian": "MENA", "Kuwaiti": "MENA", "Emirati": "MENA",
    "Omani": "MENA", "Yemeni": "MENA", "Sudanese": "MENA",
    # Sub-Saharan Africa
    "South African": "Sub-Saharan Africa", "Nigerian": "Sub-Saharan Africa",
    "Kenyan": "Sub-Saharan Africa", "Ethiopian": "Sub-Saharan Africa",
    "Somali": "Sub-Saharan Africa", "Ugandan": "Sub-Saharan Africa",
    "Zimbabwean": "Sub-Saharan Africa", "Ghanaian": "Sub-Saharan Africa",
    "South Sudanese": "Sub-Saharan Africa", "Mauritian": "Sub-Saharan Africa",
    # Western Europe
    "British": "Western Europe", "French": "Western Europe",
    "Dutch": "Western Europe", "Belgian": "Western Europe",
    "Swiss": "Western Europe", "Irish": "Western Europe",
    "Scottish": "Western Europe", "Welsh": "Western Europe",
    "Austrian": "Western Europe", "German": "Western Europe",
    # Eastern Europe
    "Russian": "Eastern Europe", "Polish": "Eastern Europe",
    "Romanian": "Eastern Europe", "Ukrainian": "Eastern Europe",
    "Czech": "Eastern Europe", "Bulgarian": "Eastern Europe",
    "Hungarian": "Eastern Europe", "Estonian": "Eastern Europe",
    "Lithuanian": "Eastern Europe", "Albanian": "Eastern Europe",
    "Serbian": "Eastern Europe", "Croatian": "Eastern Europe",
    "Bosnian": "Eastern Europe", "Macedonian": "Eastern Europe",
    # Northern Europe
    "Swedish": "Northern Europe", "Norwegian": "Northern Europe",
    "Danish": "Northern Europe", "Finnish": "Northern Europe",
    "Icelandic": "Northern Europe",
    # Southern Europe
    "Italian": "Southern Europe", "Spanish": "Southern Europe",
    "Portuguese": "Southern Europe", "Greek": "Southern Europe",
    "Maltese": "Southern Europe", "Cypriot": "Southern Europe",
    # North America
    "American": "North America", "Canadian": "North America",
    "African American": "North America", "Asian Americans": "North America",
    "Native Americans": "North America", "Puerto Rican": "North America",
    # Latin America
    "Mexican": "Latin America", "Brazilian": "Latin America",
    "Argentinian": "Latin America", "Colombian": "Latin America",
    "Chilean": "Latin America", "Venezuelan": "Latin America",
    "Peruvian": "Latin America", "Panamanian": "Latin America",
    "Cuban": "Latin America", "Latin American": "Latin America",
    # Oceania
    "Australian": "Oceania", "New Zealander": "Oceania",
    "Fijian": "Oceania", "Samoan": "Oceania", "Tongan": "Oceania",
    "Māori": "Oceania", "Papua New Guinean": "Oceania",
}


def apply_culture_merging(df: pd.DataFrame) -> pd.DataFrame:
    """Merge duplicate culture labels and drop meta-categories."""
    df = df.copy()
    df["culture"] = df["culture"].map(lambda c: CULTURE_MERGE_MAP.get(c, c))
    before = len(df)
    df = df[~df["culture"].isin(META_CULTURES_TO_DROP)].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with meta-culture labels")
    return df


def compute_region_accuracy(
    pred_ids: List[int], gold_ids: List[int], id2label: Dict[int, str],
) -> Dict[str, float]:
    """Compute region-level accuracy by mapping predicted/gold cultures to regions."""
    correct = 0
    total = 0
    for p, g in zip(pred_ids, gold_ids):
        g_culture = id2label.get(g, "")
        p_culture = id2label.get(p, "")
        g_region = CULTURE_TO_REGION.get(g_culture, "Unknown")
        p_region = CULTURE_TO_REGION.get(p_culture, "Unknown")
        if g_region != "Unknown":
            total += 1
            if p_region == g_region:
                correct += 1
    return {"region_acc": correct / total if total > 0 else 0.0, "region_total": total}


def compute_region_topk_from_logits(
    logits: np.ndarray, gold_ids: np.ndarray, id2label: Dict[int, str],
    k_values: List[int] = [3, 5],
) -> Dict[str, float]:
    """Compute region-level top-k accuracy by summing logits per region."""
    # Build culture_id → region mapping
    regions = sorted(set(CULTURE_TO_REGION.values()))
    region2id = {r: i for i, r in enumerate(regions)}
    n_regions = len(regions)
    n_samples = logits.shape[0]
    n_classes = logits.shape[1]

    # Sum logits per region
    region_logits = np.full((n_samples, n_regions), -1e9)
    for cid in range(n_classes):
        culture = id2label.get(cid, "")
        region = CULTURE_TO_REGION.get(culture, None)
        if region and region in region2id:
            rid = region2id[region]
            region_logits[:, rid] = np.logaddexp(region_logits[:, rid], logits[:, cid])

    # Map gold to region ids
    gold_region_ids = []
    valid_mask = []
    for g in gold_ids:
        culture = id2label.get(int(g), "")
        region = CULTURE_TO_REGION.get(culture, None)
        if region and region in region2id:
            gold_region_ids.append(region2id[region])
            valid_mask.append(True)
        else:
            gold_region_ids.append(-1)
            valid_mask.append(False)

    gold_region_ids = np.array(gold_region_ids)
    valid_mask = np.array(valid_mask)

    results = {}
    valid_logits = region_logits[valid_mask]
    valid_gold = gold_region_ids[valid_mask]
    n = len(valid_gold)

    # Top-1 region
    region_preds = np.argmax(valid_logits, axis=1)
    results["region_top1_acc"] = float((region_preds == valid_gold).mean()) if n > 0 else 0.0

    for k in k_values:
        actual_k = min(k, n_regions)
        top_k = np.argsort(valid_logits, axis=1)[:, -actual_k:]
        correct = sum(1 for i in range(n) if valid_gold[i] in top_k[i])
        results[f"region_top{k}_acc"] = correct / n if n > 0 else 0.0

    return results


def strip_culture_from_text(text: str, culture: str) -> str:
    """Remove culture/country mentions from norm text for Stage 2.

    Strips the demonym, its plural, and the country name so the model
    can't use the culture name as a shortcut for classification.
    """
    culture_lower = culture.lower().strip()

    # Build list of terms to remove: demonym, plural, country name
    terms = [culture_lower]

    # Add plural (e.g., "German" → "Germans")
    if culture_lower.endswith("s"):
        terms.append(culture_lower)  # already plural
    elif culture_lower.endswith("sh") or culture_lower.endswith("ch") or culture_lower.endswith("ese"):
        terms.append(culture_lower)  # no simple plural (British, French, Chinese)
    else:
        terms.append(culture_lower + "s")  # Egyptians, Australians, etc.

    # Add country name from mapping
    country = DEMONYM_TO_COUNTRY.get(culture_lower, "")
    if country:
        terms.append(country)

    # Also handle multi-word cultures directly
    # e.g., "South Korean" → also strip "South Korea"
    terms = list(set(t for t in terms if t))

    # Sort longest first to avoid partial replacements
    terms.sort(key=len, reverse=True)

    result = text
    for term in terms:
        # Remove "In [term], " at the start
        result = re.sub(r"(?i)^in\s+" + re.escape(term) + r"\s*,\s*", "", result)
        # Remove "In [term] " at the start (no comma)
        result = re.sub(r"(?i)^in\s+" + re.escape(term) + r"\s+", "", result)
        # Remove "[term] " at the start of sentence
        result = re.sub(r"(?i)^" + re.escape(term) + r"\s+", "", result)
        # Remove "in [term]" mid-sentence (with optional comma after)
        result = re.sub(r"(?i)\s+in\s+" + re.escape(term) + r"\s*,?\s*", " ", result)
        # Remove standalone mentions mid-sentence
        result = re.sub(r"(?i)\b" + re.escape(term) + r"\b", "", result)

    # Clean up artifacts: double spaces, leading/trailing spaces, orphaned commas
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r"^,\s*", "", result).strip()
    result = re.sub(r"\s,", ",", result)

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]

    return result


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
    """Build culture classification dataset from pre-split pairs (norm rows only).
    Strips culture/country names from text so the model can't use them as shortcuts."""
    def pairs_to_norm_rows(df):
        result = pd.DataFrame({
            "text": df["norm"].astype(str).str.strip(),
            "culture": df["culture"].astype(str).str.strip(),
        })
        result = result.dropna(subset=["text", "culture"])
        result = result[(result["text"] != "") & (result["culture"] != "")]

        # Strip culture mentions from text
        result["text"] = result.apply(
            lambda row: strip_culture_from_text(row["text"], row["culture"]), axis=1
        )
        result = result[result["text"] != ""].reset_index(drop=True)
        return result

    train_df = pairs_to_norm_rows(train_pairs)
    test_df = pairs_to_norm_rows(test_pairs)

    # Show examples of stripping
    print("Culture name stripping examples (Stage 2):")
    for i in range(min(5, len(test_df))):
        orig = test_pairs.iloc[i]["norm"]
        stripped = test_df.iloc[i]["text"]
        culture = test_df.iloc[i]["culture"]
        if orig != stripped:
            print(f"  [{culture}] {orig[:70]}...")
            print(f"       → {stripped[:70]}...")
            break

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


class WeightedTrainer(Trainer):
    """Trainer with weighted cross-entropy loss for imbalanced classification."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_eval_encoder(
    model_name: str,
    train_data: SplitData,
    output_dir: str,
    num_labels: int,
    epochs: int,
    batch_size: int,
    max_length: int,
    seed: int,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 1,
    class_weights: torch.Tensor = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=50,
        report_to=[],
        seed=seed,
    )

    TrainerCls = WeightedTrainer if class_weights is not None else Trainer
    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        **({"class_weights": class_weights} if class_weights is not None else {}),
    )

    trainer.train()
    pred_out = trainer.predict(test_ds)
    logits = pred_out.predictions
    preds = np.argmax(logits, axis=1)
    labels = np.array(train_data.test_labels)
    return preds, labels, logits


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
    topk: int = 1,
    learning_rate: float = 2e-4,
) -> Tuple[List[str], List[str], List[List[str]]]:
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
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="no",
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
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    trainer.train()
    pred_out = trainer.predict(test_ds)
    pred_text = tokenizer.batch_decode(pred_out.predictions, skip_special_tokens=True)
    pred_text = [p.strip() for p in pred_text]

    # Top-k beam search predictions
    topk_preds = []
    if topk > 1:
        model.eval()
        device = next(model.parameters()).device
        raw_test_texts = [f"{prefix} {t}" for t in test_texts]
        for i in range(0, len(raw_test_texts), batch_size):
            batch_txts = raw_test_texts[i:i + batch_size]
            inputs = tokenizer(batch_txts, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_input_length).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=topk,
                    num_return_sequences=topk,
                    max_length=max_target_length,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for j in range(0, len(decoded), topk):
                topk_preds.append([d.strip() for d in decoded[j:j + topk]])

    return pred_text, test_label_texts, topk_preds


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


def compute_topk_accuracy(logits: np.ndarray, labels: np.ndarray, k_values: List[int] = [3, 5, 10]) -> Dict[str, float]:
    """Compute top-k accuracy from encoder logits."""
    results = {}
    n = len(labels)
    for k in k_values:
        actual_k = min(k, logits.shape[1])
        top_k_preds = np.argsort(logits, axis=1)[:, -actual_k:]
        correct = sum(1 for i in range(n) if labels[i] in top_k_preds[i])
        results[f"top{k}_acc"] = correct / n
    return results


def compute_t5_topk_accuracy(topk_preds: List[List[str]], gold_labels: List[str], k_values: List[int] = [3, 5, 10]) -> Dict[str, float]:
    """Compute top-k accuracy from T5 beam search text predictions."""
    results = {}
    n = len(gold_labels)
    for k in k_values:
        correct = 0
        for i in range(n):
            preds_k = [normalize_label(p) for p in topk_preds[i][:k]]
            if normalize_label(gold_labels[i]) in preds_k:
                correct += 1
        results[f"top{k}_acc"] = correct / n
    return results


def slugify(model_name: str) -> str:
    return model_name.replace("/", "__")


def main():
    parser = argparse.ArgumentParser(description="Two-level norm classification pipeline (v2 - pre-split)")
    parser.add_argument("--train_csv", required=True, help="Path to train pairs CSV (culture, original_norm, norm, generic)")
    parser.add_argument("--test_csv", required=True, help="Path to test pairs CSV (same columns)")
    parser.add_argument("--output_dir", default="two_level_results_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4, help="Epochs for Stage 1 (binary)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for Stage 1")
    parser.add_argument("--stage2_epochs", type=int, default=8, help="Epochs for Stage 2 (culture classification)")
    parser.add_argument("--stage2_batch_size", type=int, default=64, help="Batch size for Stage 2")
    parser.add_argument("--stage2_lr", type=float, default=2e-5, help="Learning rate for Stage 2 encoder models")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--weighted_loss", action="store_true", help="Use inverse-frequency weighted CE loss for Stage 2")
    parser.add_argument("--skip_stage1", action="store_true", help="Skip Stage 1 (binary), only run Stage 2 (culture)")
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

    print(f"Train pairs (raw): {len(train_pairs)} ({train_pairs['culture'].nunique()} cultures)")
    print(f"Test pairs (raw):  {len(test_pairs)} ({test_pairs['culture'].nunique()} cultures)")

    # Apply culture merging and drop meta-categories
    print("Applying culture label merging...")
    train_pairs = apply_culture_merging(train_pairs)
    test_pairs = apply_culture_merging(test_pairs)
    print(f"Train pairs (merged): {len(train_pairs)} ({train_pairs['culture'].nunique()} cultures)")
    print(f"Test pairs (merged):  {len(test_pairs)} ({test_pairs['culture'].nunique()} cultures)")

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

    # Compute class weights for weighted loss
    class_weights = None
    if args.weighted_loss:
        counts = np.bincount(norm_data.train_labels, minlength=len(label2id)).astype(np.float32)
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.sum() * len(label2id)  # normalize: mean weight = 1
        class_weights = torch.tensor(weights, dtype=torch.float32)
        print(f"Weighted loss enabled: min_weight={weights.min():.2f}, max_weight={weights.max():.2f}")

    summary_rows = []
    labelwise_all_rows = []

    for model_name in args.models:
        print("\n" + "=" * 90)
        print(f"MODEL: {model_name}")
        print("=" * 90)
        model_slug = slugify(model_name)

        binary_acc = np.nan
        norm_acc = np.nan
        topk_results = {}
        region_results = {}
        stage1_status = "ok"
        stage2_status = "ok"

        if args.skip_stage1:
            print("Skipping Stage 1 (--skip_stage1 set)")
            stage1_status = "skipped"
        else:
            try:
                if "flan-t5" in model_name.lower() or model_name.lower().startswith("t5"):
                    train_lbl_txt = ["norm" if y == 1 else "fact" for y in binary_data.train_labels]
                    test_lbl_txt = ["norm" if y == 1 else "fact" for y in binary_data.test_labels]
                    pred_txt, gold_txt, _ = train_eval_t5(
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
                    pred, gold, _ = train_eval_encoder(
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
                pred_txt, gold_txt, topk_preds = train_eval_t5(
                    model_name=model_name,
                    train_texts=norm_data.train_texts,
                    train_label_texts=train_lbl_txt,
                    test_texts=norm_data.test_texts,
                    test_label_texts=test_lbl_txt,
                    output_dir=os.path.join(args.output_dir, f"tmp_{model_slug}_stage2"),
                    epochs=args.stage2_epochs,
                    batch_size=args.stage2_batch_size,
                    max_input_length=args.max_length,
                    max_target_length=16,
                    seed=args.seed,
                    prefix="classify norm culture label:",
                    topk=10,
                    learning_rate=args.stage2_lr * 10,  # T5 uses higher LR
                )
                norm_pred_ids = [label2id.get(p, -1) for p in pred_txt]
                norm_gold_ids = [label2id[g] for g in gold_txt]
                norm_acc = accuracy_score(norm_gold_ids, norm_pred_ids)
                topk_results = compute_t5_topk_accuracy(topk_preds, test_lbl_txt, k_values=[3, 5, 10])
            else:
                pred, gold, logits = train_eval_encoder(
                    model_name=model_name,
                    train_data=norm_data,
                    output_dir=os.path.join(args.output_dir, f"tmp_{model_slug}_stage2"),
                    num_labels=len(label2id),
                    epochs=args.stage2_epochs,
                    batch_size=args.stage2_batch_size,
                    max_length=args.max_length,
                    seed=args.seed,
                    learning_rate=args.stage2_lr,
                    gradient_accumulation_steps=args.gradient_accumulation_steps,
                    class_weights=class_weights,
                )
                norm_pred_ids = pred.tolist()
                norm_gold_ids = gold.tolist()
                norm_acc = accuracy_score(norm_gold_ids, norm_pred_ids)
                topk_results = compute_topk_accuracy(logits, gold, k_values=[3, 5, 10])

            lw = labelwise_accuracy(norm_gold_ids, norm_pred_ids, id2label)
            lw["model"] = model_name
            labelwise_all_rows.append(lw)
            lw.to_csv(os.path.join(args.output_dir, f"labelwise_{model_slug}.csv"), index=False)

            # Region accuracy
            region_results = compute_region_accuracy(norm_pred_ids, norm_gold_ids, id2label)
            if "flan-t5" in model_name.lower() or model_name.lower().startswith("t5"):
                # T5: compute region top-k from beam search predictions
                if topk_preds:
                    n = len(test_lbl_txt)
                    for k in [3, 5]:
                        correct = 0
                        total = 0
                        for i in range(n):
                            gold_region = CULTURE_TO_REGION.get(test_lbl_txt[i], "Unknown")
                            if gold_region == "Unknown":
                                continue
                            total += 1
                            pred_regions = [CULTURE_TO_REGION.get(p.strip(), "") for p in topk_preds[i][:k]]
                            if gold_region in pred_regions:
                                correct += 1
                        region_results[f"region_top{k}_acc"] = correct / total if total > 0 else 0.0
            else:
                region_topk = compute_region_topk_from_logits(logits, gold, id2label, k_values=[3, 5])
                region_results.update(region_topk)

            topk_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(topk_results.items()))
            region_str = "  ".join(f"{k}={v:.4f}" for k, v in sorted(region_results.items()) if k != "region_total")
            print(f"Stage 2 (Culture) accuracy: top-1={norm_acc:.4f}  {topk_str}")
            print(f"Stage 2 (Region)  accuracy: {region_str}")
        except Exception as e:
            stage2_status = f"failed: {e}"
            print(f"Stage 2 failed for {model_name}: {e}")

        summary_rows.append(
            {
                "model": model_name,
                "binary_test_accuracy": binary_acc,
                "norm_label_test_accuracy": norm_acc,
                "norm_top3_accuracy": topk_results.get("top3_acc", np.nan),
                "norm_top5_accuracy": topk_results.get("top5_acc", np.nan),
                "norm_top10_accuracy": topk_results.get("top10_acc", np.nan),
                "region_top1_accuracy": region_results.get("region_top1_acc", region_results.get("region_acc", np.nan)),
                "region_top3_accuracy": region_results.get("region_top3_acc", np.nan),
                "region_top5_accuracy": region_results.get("region_top5_acc", np.nan),
                "binary_train_size": len(binary_data.train_texts),
                "binary_test_size": len(binary_data.test_texts),
                "norm_train_size": len(norm_data.train_texts),
                "norm_test_size": len(norm_data.test_texts),
                "n_norm_labels": len(label2id),
                "stage1_status": stage1_status,
                "stage2_status": stage2_status,
                "stage1_epochs": args.epochs,
                "stage1_batch_size": args.batch_size,
                "stage2_epochs": args.stage2_epochs,
                "stage2_batch_size": args.stage2_batch_size,
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
    display_cols = ["model", "binary_test_accuracy", "norm_label_test_accuracy", "norm_top5_accuracy", "norm_top10_accuracy", "region_top1_accuracy", "region_top3_accuracy", "stage1_status", "stage2_status"]
    display_df = summary_df[display_cols].copy()
    print(display_df.to_string(index=False))

    print("\nDone.")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
