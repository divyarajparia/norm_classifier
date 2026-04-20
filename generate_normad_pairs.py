"""
Generate paired statements (norm + generic) for NormAd dataset using OpenAI API.
Same approach as generate_factual_negatives.py but for NormAd norms.

Input: normad_cleaned.csv (from analyze_normad.py)
Output: normad_pairs_openai.csv (columns: culture, original_norm, norm, generic)

Resumable, parallel, uses .env for API key.
"""

import csv
import os
import re
import time
import threading
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# ── CONFIG ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL_NAME = "gpt-4o-mini"
BATCH_SIZE = 20
NUM_WORKERS = 5

INPUT_FILE = "normad_cleaned.csv"
OUTPUT_FILE = "normad_pairs_openai.csv"
LOG_FILE = "normad_generation_log.txt"

write_lock = threading.Lock()


def build_prompt(batch_rows):
    """Build prompt for a batch of NormAd norms."""
    inputs = []
    for i, row in enumerate(batch_rows, 1):
        subaxis = row.get("subaxis", "").replace("_", " ") if row.get("subaxis") else ""
        value = row.get("value", "") if row.get("value") else ""
        parts = [f"[{i}] Culture: {row['culture']}", f"Norm: {row['norm_text']}"]
        if subaxis:
            parts.append(f"Etiquette type: {subaxis}")
        if value:
            parts.append(f"Value: {value}")
        inputs.append(" | ".join(parts))

    input_block = "\n".join(inputs)
    n = len(batch_rows)

    return f"""You are given cultural etiquette norms from the NormAd dataset. Each entry has:
- **Culture**: The demonym (e.g., Egyptian, Swedish, Colombian)
- **Norm**: A Rule-of-Thumb describing what is expected, polite, or correct in that culture
- **Etiquette type**: The category (e.g., basic etiquette, eating, visiting, gift giving)
- **Value**: The abstract value behind the norm (e.g., "Respect and modesty")

For each norm, generate exactly TWO statements:

**N (Norm):** Creatively rephrase the norm as a natural-sounding sentence about what people of that culture do or are expected to do. DO NOT just prepend "In [country]..." to the original — genuinely reword it so it reads like an observation from a cultural guide or travel book. Keep the normative meaning intact.

**G (Generic):** A factual, technical, or definitional sentence that shares at least 2 key content words with N but is completely non-normative. It could be a scientific fact, a dictionary-style definition, a historical tidbit, or an objective observation — anything that is NOT about what people do, should do, or are expected to do.

Rules:
1. Both N and G must be 10-25 words long and similar in length to each other.
2. G MUST share at least 2 key content words with N (nouns or verbs, not articles/prepositions).
3. G must NOT describe any behavior, custom, tradition, or social expectation.
4. N describes what people DO or are expected to do (normative). G describes what something IS (factual/technical).
5. N CAN use prescriptive language like "should", "must", "expected to" — norms are prescriptive by definition. G must NOT use such language.
6. Do NOT start every N with "In [Country]..." — vary your sentence structure. Use the culture's demonym naturally (e.g., "Egyptians greet...", "Visitors to Sweden are expected...", "When dining in Colombia...").

Examples:

[1] Culture: Egyptian | Norm: It is respectful to greet everyone present before starting any social interaction. | Etiquette type: basic etiquette | Value: Respect and modesty in interpersonal interactions.
[1] N: Egyptians consider it essential to personally greet every person in the room before engaging in conversation.
[1] G: A greeting is a conventional expression used upon meeting someone, varying in formality across languages.

[2] Culture: Swedish | Norm: Always finish the food on your plate during a meal to show respect. | Etiquette type: eating | Value: Respect for the host's efforts and understanding the significance of dining etiquette.
[2] N: Swedish diners are expected to finish all the food on their plate as a sign of appreciation for the host's effort.
[2] G: The average dinner plate has a diameter of approximately 10 to 12 inches and is typically made of ceramic.

[3] Culture: Australian | Norm: It is customary to open gifts in front of the giver to show appreciation. | Etiquette type: gifts | Value: Thoughtfulness and consideration in gift-giving.
[3] N: Australians typically unwrap gifts immediately in front of the giver, expressing gratitude and genuine reaction.
[3] G: Gift wrapping paper was first commercially produced in the early twentieth century by a Kansas City stationery store.

[4] Culture: Sudanese | Norm: Politely decline an offer twice before accepting it on the third instance to show humility. | Etiquette type: offering and complimenting items | Value: Graciousness and humility in social interactions.
[4] N: Sudanese guests are expected to politely decline an offer twice before graciously accepting it the third time.
[4] G: The number three appears frequently in mathematics as the first odd prime and the second smallest prime number.

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
                last = max(last, int(line.split("=")[1].split("|")[0].strip()))
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
                        "original_norm": batch_rows[j]["norm_text"],
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

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["norm_text"]).reset_index(drop=True)
    total = len(df)
    print(f"Loaded {total} norms from {df['culture'].nunique()} cultures")

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
    print(f"Estimated time: ~{max(1, num_calls * 2 // NUM_WORKERS // 60)} min\n")

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
