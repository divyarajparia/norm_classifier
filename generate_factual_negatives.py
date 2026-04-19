"""
Generate paired statements (norm + generic) using OpenAI API.
For each CultureBank norm, the model produces:
  - A norm statement (label=1): rephrased behavioral expectation
  - A generic statement (label=0): shares keywords, but non-normative

Both from the same LLM = same semantic space, same style, same length.
The model must learn the actual norm vs non-norm distinction.

Resumable: saves after every batch to generated_pairs.csv.
Tracks last completed index in generation_log.txt.
Parallel: uses 5 concurrent workers for ~5x speedup.
"""

import csv
import os
import re
import time
import threading
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = "gpt-4o-mini"
BATCH_SIZE = 20  # norms per API call (each produces 2 statements)
NUM_WORKERS = 5  # parallel API calls

OUTPUT_FILE = "generated_pairs.csv"
LOG_FILE = "generation_log.txt"

# Thread lock for file writes
write_lock = threading.Lock()

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
    """Load CultureBank, filter labels, merge Americans, cap at 800."""
    print("Loading CultureBank...")
    cb = load_dataset("SALT-NLP/CultureBank")
    df_tiktok = cb["tiktok"].to_pandas()
    df_reddit = cb["reddit"].to_pandas()
    df_cb = pd.concat([df_tiktok, df_reddit], ignore_index=True)

    df_cb = df_cb[df_cb["agreement"] >= 0.6]
    df_cb = df_cb[df_cb["actor_behavior"].notna()]
    df_cb = df_cb[df_cb["actor_behavior"].str.strip() != ""]

    df = pd.DataFrame()
    df["culture"] = df_cb["cultural group"].values
    df["actor_behavior"] = df_cb["actor_behavior"].str.strip().values
    df["context"] = df_cb["context"].fillna("").str.strip().values
    df["topic"] = df_cb["topic"].fillna("").str.strip().values

    vc = df["culture"].value_counts()
    labels_to_remove = set()
    for label, count in vc.items():
        if should_remove_label(label) or count < 10:
            labels_to_remove.add(label)
    removed_count = df[df["culture"].isin(labels_to_remove)].shape[0]
    df = df[~df["culture"].isin(labels_to_remove)].reset_index(drop=True)

    us_labels = {"American", "Americans", "United States"}
    df.loc[df["culture"].isin(us_labels), "culture"] = "American"

    american_mask = df["culture"] == "American"
    if american_mask.sum() > 800:
        american_df = df[american_mask].sample(n=800, random_state=42)
        df = pd.concat([df[~american_mask], american_df], ignore_index=True)

    print(f"Removed {removed_count} norms ({len(labels_to_remove)} labels)")
    print(f"Final: {len(df)} norms, {df['culture'].nunique()} labels")
    return df


def build_prompt(batch_rows):
    """Build prompt for a batch of norms."""
    inputs = []
    for i, row in enumerate(batch_rows, 1):
        ctx = f" | Context: {row['context']}" if row['context'] else ""
        topic = f" | Topic: {row['topic']}" if row['topic'] else ""
        inputs.append(f"[{i}] Culture: {row['culture']} | Norm: {row['actor_behavior']}{ctx}{topic}")

    input_block = "\n".join(inputs)
    n = len(batch_rows)

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
""" + "\n".join(f"[{i}] N: ...\n[{i}] G: ..." for i in range(1, n + 1))


def parse_response(text, batch_size):
    """Parse response into (norm, generic) pairs."""
    pairs = {}
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"\[(\d+)\]\s*([NG]):\s*(.+)", line)
        if match:
            idx = int(match.group(1))
            stmt_type = match.group(2)
            stmt_text = match.group(3).strip()
            if idx not in pairs:
                pairs[idx] = {}
            pairs[idx][stmt_type] = stmt_text

    results = []
    for i in range(1, batch_size + 1):
        if i in pairs and "N" in pairs[i] and "G" in pairs[i]:
            results.append((pairs[i]["N"], pairs[i]["G"]))
        else:
            results.append(None)

    return results


def get_last_completed_index():
    if not os.path.exists(LOG_FILE):
        return -1
    last = -1
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("COMPLETED_INDEX="):
                last = int(line.split("=")[1].split("|")[0].strip())
    return last


def log_progress(index, total, culture, status):
    with write_lock:
        with open(LOG_FILE, "a") as f:
            f.write(f"COMPLETED_INDEX={index} | {index+1}/{total} | culture={culture} | {status}\n")


def save_pairs(pairs_to_save):
    with write_lock:
        file_exists = os.path.exists(OUTPUT_FILE)
        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["culture", "original_norm", "norm", "generic"])
            if not file_exists:
                writer.writeheader()
            for row in pairs_to_save:
                writer.writerow(row)


def process_batch(client, batch_rows, batch_start, batch_end, total):
    """Process a single batch — called by worker threads."""
    prompt = build_prompt(batch_rows)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            response_text = response.choices[0].message.content
            pairs = parse_response(response_text, len(batch_rows))

            pairs_to_save = []
            success = 0
            for j, pair in enumerate(pairs):
                if pair is not None:
                    norm_text, generic_text = pair
                    pairs_to_save.append({
                        "culture": batch_rows[j]["culture"],
                        "original_norm": batch_rows[j]["actor_behavior"],
                        "norm": norm_text,
                        "generic": generic_text,
                    })
                    success += 1

            if pairs_to_save:
                save_pairs(pairs_to_save)

            log_progress(batch_end - 1, total, batch_rows[-1]["culture"],
                         f"OK {success}/{len(batch_rows)}")
            print(f"  [{batch_end}/{total}] +{success} pairs | {batch_rows[0]['culture']}")
            return success

        except Exception as e:
            err = str(e)[:120]
            if "429" in err or "rate_limit" in err.lower():
                wait = 10 * (attempt + 1)
                print(f"  Rate limited at {batch_start}, waiting {wait}s (attempt {attempt+1})...")
                time.sleep(wait)
                continue
            else:
                print(f"  ERROR at {batch_start}: {err}")
                log_progress(batch_end - 1, total, batch_rows[-1]["culture"],
                             f"ERROR: {err[:60]}")
                return 0

    return 0


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    df = load_and_filter_norms()
    total = len(df)

    last_done = get_last_completed_index()
    start = last_done + 1
    if start > 0:
        print(f"\nResuming from index {start} (already completed {start}/{total})")
    else:
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
        print(f"\nStarting fresh generation for {total} norms")

    num_calls = (total - start + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Batch size: {BATCH_SIZE} | Workers: {NUM_WORKERS} | API calls: {num_calls}")
    print(f"Estimated time: ~{num_calls * 2 // NUM_WORKERS // 60} min\n")

    # Build all batches
    batches = []
    i = start
    while i < total:
        batch_end = min(i + BATCH_SIZE, total)
        batch_df = df.iloc[i:batch_end]
        batch_rows = batch_df.to_dict("records")
        batches.append((batch_rows, i, batch_end))
        i = batch_end

    # Process in parallel
    total_success = 0
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for batch_rows, batch_start, batch_end in batches:
            future = executor.submit(process_batch, client, batch_rows,
                                     batch_start, batch_end, total)
            futures[future] = (batch_start, batch_end)

        for future in as_completed(futures):
            result = future.result()
            total_success += result

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
