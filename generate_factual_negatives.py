"""
Generate paired statements (norm + generic) using Gemini API.
For each CultureBank norm, Gemini produces:
  - A norm statement (label=1): rephrased behavioral expectation
  - A generic statement (label=0): shares keywords, but non-normative

Both from Gemini = same semantic space, same style, same length.
The model must learn the actual norm vs non-norm distinction.

Resumable: saves after every batch to generated_pairs.csv.
Tracks last completed index in generation_log.txt.
"""

import csv
import os
import re
import time
import pandas as pd
from google import genai
from datasets import load_dataset

# ── CONFIG ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
MODEL_NAME = "gemini-2.5-flash"
BATCH_SIZE = 10  # norms per API call (each produces 2 statements)
RATE_LIMIT_DELAY = 5  # seconds between API calls

OUTPUT_FILE = "generated_pairs.csv"
LOG_FILE = "generation_log.txt"

# ── LABEL FILTERING ─────────────────────────────────────────────────────────
OTHER_KEYWORDS = [
    "people", "non-", "expat", "immigrant", "family", "lgbtq", "gay",
    "queer", "vegan", "vegetarian", "gen z", "boomer", "millennial",
    "student", "global",
]
RELIGION_KEYWORDS = [
    "muslim", "christian", "jewish", "jew", "hindu", "buddhist",
    "mormon", "amish", "sikh", "catholic", "orthodox",
]
EXTRA_REMOVE = {"parent", "english speakers", "spanish speakers"}


def should_remove_label(label):
    c = label.lower().strip()
    if c in EXTRA_REMOVE:
        return True
    if any(kw in c for kw in OTHER_KEYWORDS):
        return True
    if any(kw in c for kw in RELIGION_KEYWORDS):
        return True
    return False


def load_and_filter_norms():
    """Load CultureBank, filter labels, merge Americans, cap at 800.
    Returns DataFrame with columns: culture, actor_behavior, context, topic.
    """
    print("Loading CultureBank...")
    cb = load_dataset("SALT-NLP/CultureBank")
    df_tiktok = cb["tiktok"].to_pandas()
    df_reddit = cb["reddit"].to_pandas()
    df_cb = pd.concat([df_tiktok, df_reddit], ignore_index=True)

    # Filter high agreement + non-empty actor_behavior
    df_cb = df_cb[df_cb["agreement"] >= 0.6]
    df_cb = df_cb[df_cb["actor_behavior"].notna()]
    df_cb = df_cb[df_cb["actor_behavior"].str.strip() != ""]

    # Extract columns we need
    df = pd.DataFrame()
    df["culture"] = df_cb["cultural group"].values
    df["actor_behavior"] = df_cb["actor_behavior"].str.strip().values
    df["context"] = df_cb["context"].fillna("").str.strip().values
    df["topic"] = df_cb["topic"].fillna("").str.strip().values

    # Remove OTHER, RELIGION, count < 10
    vc = df["culture"].value_counts()
    labels_to_remove = set()
    for label, count in vc.items():
        if should_remove_label(label) or count < 10:
            labels_to_remove.add(label)
    removed_count = df[df["culture"].isin(labels_to_remove)].shape[0]
    df = df[~df["culture"].isin(labels_to_remove)].reset_index(drop=True)

    # Merge American/Americans/United States -> American
    us_labels = {"American", "Americans", "United States"}
    df.loc[df["culture"].isin(us_labels), "culture"] = "American"

    # Cap American at 800
    american_mask = df["culture"] == "American"
    if american_mask.sum() > 800:
        american_df = df[american_mask].sample(n=800, random_state=42)
        df = pd.concat([df[~american_mask], american_df], ignore_index=True)

    print(f"Removed {removed_count} norms ({len(labels_to_remove)} labels)")
    print(f"Final: {len(df)} norms, {df['culture'].nunique()} labels")
    return df


def build_prompt(batch_rows):
    """Build Gemini prompt for a batch of norms.
    Each row is a dict with: culture, actor_behavior, context, topic.
    """
    # Build the input list
    inputs = []
    for i, row in enumerate(batch_rows, 1):
        ctx = f" | Context: {row['context']}" if row['context'] else ""
        topic = f" | Topic: {row['topic']}" if row['topic'] else ""
        inputs.append(f"[{i}] Culture: {row['culture']} | Norm: {row['actor_behavior']}{ctx}{topic}")

    input_block = "\n".join(inputs)

    return f"""You are given cultural norms. For each norm, generate exactly TWO statements:

**N (Norm):** Rephrase the norm as a complete sentence describing what people of that culture do or are expected to do. It should read like an observation from a cultural guide. Keep the normative meaning — it must describe a behavioral expectation, practice, or value.

**G (Generic):** Generate a factual, technical, or definitional sentence that reuses at least 2 key content words from the norm but describes something completely non-normative. It could be a scientific fact, a technical definition, a historical date, or an objective observation — anything that is NOT about what people do or should do.

Rules:
1. Both N and G must be 10-25 words long and similar in length to each other.
2. G MUST share at least 2 key content words with N.
3. G must NOT describe any behavior, custom, tradition, or social expectation.
4. N describes what people DO (normative). G describes what something IS (factual/technical).
5. N CAN use prescriptive language like "should", "must", "expected to" — norms are prescriptive by definition. G must NOT use such language — generics don't prescribe behavior.

Example:
[1] Culture: American | Norm: tip as an expression of gratitude for good service | Context: in restaurants | Topic: Social Norms
[1] N: American diners often leave a monetary tip at restaurants as a gesture of gratitude for attentive service.
[1] G: The word "tip" derives from eighteenth-century coffeehouse slang and originally stood for "To Insure Promptness."

[2] Culture: Japanese | Norm: remove their shoes before entering homes | Context: at home | Topic: Daily Life
[2] N: In Japan, residents and visitors typically remove their shoes at the entryway before stepping inside a home.
[2] G: Modern shoe soles are manufactured by entering heated rubber compounds into precision-engineered industrial molds.

---

{input_block}

Output ONLY in this exact format, nothing else:
[1] N: ...
[1] G: ...
[2] N: ...
[2] G: ...
"""


def parse_response(text, batch_size):
    """Parse Gemini response into (norm, generic) pairs.
    Returns list of (norm_text, generic_text) tuples.
    """
    pairs = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match [idx] N: text or [idx] G: text
        match = re.match(r"\[(\d+)\]\s*([NG]):\s*(.+)", line)
        if match:
            idx = int(match.group(1))
            stmt_type = match.group(2)
            stmt_text = match.group(3).strip()
            if idx not in pairs:
                pairs[idx] = {}
            pairs[idx][stmt_type] = stmt_text

    # Build ordered list of complete pairs
    results = []
    for i in range(1, batch_size + 1):
        if i in pairs and "N" in pairs[i] and "G" in pairs[i]:
            results.append((pairs[i]["N"], pairs[i]["G"]))
        else:
            results.append(None)  # incomplete pair

    return results


def get_last_completed_index():
    """Read the last completed index from log file."""
    if not os.path.exists(LOG_FILE):
        return -1
    last = -1
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("COMPLETED_INDEX="):
                last = int(line.split("=")[1])
    return last


def log_progress(index, total, culture, status):
    """Append progress to log file."""
    with open(LOG_FILE, "a") as f:
        f.write(f"COMPLETED_INDEX={index} | {index+1}/{total} | culture={culture} | {status}\n")


def save_pairs(pairs_to_save):
    """Append pairs to output CSV."""
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["culture", "original_norm", "norm", "generic"])
        if not file_exists:
            writer.writeheader()
        for row in pairs_to_save:
            writer.writerow(row)


def main():
    client = genai.Client(api_key=GEMINI_API_KEY)

    # Load and filter norms
    df = load_and_filter_norms()
    total = len(df)

    # Check resume point
    last_done = get_last_completed_index()
    start = last_done + 1
    if start > 0:
        print(f"\nResuming from index {start} (already completed {start}/{total})")
    else:
        # Fresh start — clear output file if exists
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        print(f"\nStarting fresh generation for {total} norms")

    print(f"Batch size: {BATCH_SIZE} norms per API call")
    print(f"Estimated API calls: {(total - start + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"Estimated time: ~{((total - start + BATCH_SIZE - 1) // BATCH_SIZE) * RATE_LIMIT_DELAY // 60} min\n")

    # Process in batches
    i = start
    while i < total:
        batch_end = min(i + BATCH_SIZE, total)
        batch_df = df.iloc[i:batch_end]
        batch_rows = batch_df.to_dict("records")

        prompt = build_prompt(batch_rows)

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            pairs = parse_response(response.text, len(batch_rows))

            # Save successful pairs
            pairs_to_save = []
            success = 0
            for j, pair in enumerate(pairs):
                row_idx = i + j
                if pair is not None:
                    norm_text, generic_text = pair
                    pairs_to_save.append({
                        "culture": batch_rows[j]["culture"],
                        "original_norm": batch_rows[j]["actor_behavior"],
                        "norm": norm_text,
                        "generic": generic_text,
                    })
                    success += 1
                else:
                    # Log failed parse — will be skipped
                    print(f"    WARNING: Failed to parse pair for index {row_idx}")

            if pairs_to_save:
                save_pairs(pairs_to_save)

            # Log progress
            log_progress(batch_end - 1, total, batch_rows[-1]["culture"],
                         f"OK {success}/{len(batch_rows)}")
            print(f"  [{batch_end}/{total}] +{success} pairs | {batch_rows[0]['culture']}")

            i = batch_end

        except Exception as e:
            err = str(e)[:120]
            print(f"  ERROR at index {i}: {err}")
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print("  Rate limited — waiting 60s before retry...")
                time.sleep(60)
                continue  # retry same batch
            else:
                # Skip this batch on other errors
                log_progress(batch_end - 1, total, batch_rows[-1]["culture"],
                             f"ERROR: {err[:60]}")
                i = batch_end

        time.sleep(RATE_LIMIT_DELAY)

    # Final summary
    if os.path.exists(OUTPUT_FILE):
        df_out = pd.read_csv(OUTPUT_FILE)
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"  Total pairs: {len(df_out)}")
        print(f"  Cultures: {df_out['culture'].nunique()}")
        print(f"  Norm avg words: {df_out['norm'].str.split().str.len().mean():.1f}")
        print(f"  Generic avg words: {df_out['generic'].str.split().str.len().mean():.1f}")
        print(f"  Saved to: {OUTPUT_FILE}")
        print(f"{'='*60}")
    else:
        print("\nNo output generated.")


if __name__ == "__main__":
    main()
