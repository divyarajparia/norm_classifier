# Norm Classifier

Binary text classifier: **social norm** (label=1) vs **generic statement** (label=0), using fine-tuned BERT (`bert-base-uncased`).

> A norm is a behavioral expectation, practice, or value that members of a culture follow — describing WHAT PEOPLE DO or SHOULD DO. A generic sentence describes factual information, observations, or general statements without prescribing behavior.

## Data Generation Approach

For each CultureBank norm, **Gemini generates a paired norm + generic statement**. Both come from the same model, sharing keywords and length, so the classifier must learn actual norm semantics — not style, length, or vocabulary shortcuts.

| Label | What | Example |
|-------|------|---------|
| 1 (norm) | Behavioral expectation (Gemini-rephrased) | "Germans typically adhere to strict recycling rules, separating household waste into multiple bins." |
| 0 (generic) | Factual/technical statement (Gemini-generated, shares keywords) | "The process of recycling waste can significantly reduce landfill volume and conserve natural resources." |

**Columns fed to Gemini**: `actor_behavior`, `cultural group`, `context`, `topic`

## Dataset Decisions

- **Source**: [CultureBank](https://huggingface.co/datasets/SALT-NLP/CultureBank) (TikTok + Reddit, agreement >= 0.6)
- **Labels removed**: OTHER ("people", "non-X", "family", "global"), RELIGION (Muslim, Christian, Jewish, etc.), count < 10
- **Labels kept as-is**: country demonyms, sub-national (Californians, Texans), ethnic (African Americans), continental (European, Asian)
- **American/Americans**: merged into "American", capped at 800 samples
- **Scottish/Welsh/English**: kept separate (not merged into UK)
- **Gemini model**: `gemini-2.5-flash`
- **Split**: 70/15/15 stratified on label

## Pipeline

```bash
conda activate nlp_env
pip install -r requirements.txt

# 1. Generate paired statements via Gemini (resumable, logs to generation_log.txt)
python generate_factual_negatives.py

# 2. Assemble train/val/test CSVs
python prepare_dataset_v2.py

# 3. Train BERT classifier (3 epochs)
python train_bert.py
```

## Project Structure

```
.
├── generate_factual_negatives.py  # Gemini paired generation (norm + generic)
├── prepare_dataset_v2.py          # Assemble and split dataset
├── train_bert.py                  # BERT fine-tuning and evaluation
├── explore_dataset.py             # EDA with plots
├── prepare_dataset.py             # v1 pipeline (AG News, kept for reference)
├── requirements.txt               # Python dependencies
├── generated_pairs.csv            # Gemini output: culture, original_norm, norm, generic
├── generation_log.txt             # Resume log for Gemini generation
├── train.csv / val.csv / test.csv # Final dataset splits
├── norm_classifier_bert/          # Saved model (git-ignored)
└── plot_*.png                     # EDA plots
```

## What's in `.gitignore`

Only source code, `generated_pairs.csv`, the PDF spec, and `requirements.txt` are tracked. Everything else is generated and git-ignored:

| Pattern | What it excludes |
|---------|-----------------|
| `norm_classifier_bert/` | Saved BERT model weights (~440 MB) |
| `*.safetensors`, `*.pt`, `*.pth`, `*.bin` | Other model checkpoint formats |
| `*.csv` (except `generated_pairs.csv`) | `train.csv`, `val.csv`, `test.csv`, `test_gemini_output.csv` |
| `generation_log.txt` | Gemini generation resume log |
| `*.png` | All generated plots (`plot_*.png`) |
| `__pycache__/`, `*.py[cod]` | Python bytecode |
| `venv/`, `env/`, `.venv/` | Virtual environments |
| `.ipynb_checkpoints/` | Jupyter autosaves |
| `.vscode/`, `.idea/` | IDE settings |
| `.DS_Store`, `Thumbs.db` | OS junk |
| `.claude/` | Claude Code settings |

## Replicating after cloning

After cloning, the repo has source code + `generated_pairs.csv` (the Gemini-generated data). CultureBank is loaded from HuggingFace at runtime — no manual download needed. To get the full working project:

```bash
# 1. Install dependencies
conda activate nlp_env          # or create a new venv
pip install -r requirements.txt

# 2. Set your Gemini API key (only needed if re-running generation)
export GEMINI_API_KEY="your-key-here"

# 3. Assemble train/val/test splits from the tracked generated_pairs.csv
python prepare_dataset_v2.py
#    → produces: train.csv, val.csv, test.csv

# 4. Train the BERT classifier (3 epochs)
python train_bert.py
#    → produces: norm_classifier_bert/ (saved model)

# 5. (Optional) Explore the dataset and generate plots
python explore_dataset.py
#    → produces: plot_source_dist.png, plot_topics.png, plot_cultures.png, plot_agreement.png
```

To regenerate `generated_pairs.csv` from scratch (~30-40 min, resumable):
```bash
export GEMINI_API_KEY="your-key-here"
python generate_factual_negatives.py
```

## Bonus Task

If a sentence is a norm, which culture does it belong to? (norm → culture label: USA, India, Japan, universal, etc.)
