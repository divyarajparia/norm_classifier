"""
Generate culturally-authentic synthetic norms for minority cultures using LLM world knowledge.

Unlike oversampling (which rephrases existing norms), this script asks the LLM to generate
BRAND NEW norms based on its knowledge of each culture — covering food, greetings, family,
business, celebrations, dress, hospitality, etc.

For each synthetic norm, a matching generic (factual) statement is also generated
to maintain the pairs format needed by the pipeline.

Input:  train_merged.csv
Output: train_merged_synthetic.csv

Only generates for cultures below the median count. Test set is never touched.
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
NUM_WORKERS = 5

INPUT_FILE = "train_merged.csv"
OUTPUT_FILE = "train_merged_synthetic.csv"
SYNTHETIC_FILE = "synthetic_knowledge_pairs.csv"
LOG_FILE = "synthetic_generation_log.txt"

write_lock = threading.Lock()


def build_system_prompt():
    """System prompt that primes the LLM for culturally-grounded norm generation."""
    return """You are a cultural anthropologist and etiquette expert with deep knowledge of social norms, customs, and behavioral expectations across world cultures.

Your task is to generate culturally-specific social norms — statements that describe what people of a particular culture DO, SHOULD DO, or ARE EXPECTED TO DO in everyday life.

A good cultural norm is:
- SPECIFIC to that culture, not a universal human behavior (e.g., "Japanese diners slurp noodles to show appreciation" is specific; "People eat food when hungry" is universal)
- OBSERVABLE — it describes a concrete behavior, not an abstract value
- ACCURATE — it reflects real practices that members of that culture would recognize as true
- DIVERSE — covering different domains of life: dining, greetings, hospitality, gift-giving, family, business, celebrations, dress, public behavior, religious customs, communication style, etc.

You also know the difference between a norm and a generic factual statement:
- NORM: "Koreans are expected to use both hands when receiving something from an elder" (prescriptive, about behavior)
- GENERIC: "The Korean peninsula extends approximately 1,100 km from north to south" (factual, not about behavior)"""


def build_generation_prompt(culture, num_to_generate, existing_norms):
    """Build the user prompt for generating new norms for a specific culture."""
    # Format existing norms as few-shot examples
    example_lines = []
    for i, norm in enumerate(existing_norms, 1):
        example_lines.append(f"  {i}. {norm}")
    examples_block = "\n".join(example_lines)

    return f"""Generate {num_to_generate} NEW culturally-authentic social norms for **{culture}** culture.

Here are some existing norms we already have for {culture} (DO NOT repeat these — generate completely new ones):
{examples_block}

Requirements for each norm:
1. It must be a REAL cultural practice or expectation specific to {culture} people — draw on your knowledge of their etiquette, dining customs, social hierarchies, hospitality traditions, communication styles, family dynamics, religious influences, business culture, celebrations, and daily habits.
2. It must describe what {culture} people DO, SHOULD DO, or ARE EXPECTED TO DO — use prescriptive language naturally ("expected to", "considered polite to", "typically", "should", "must").
3. It must be SPECIFIC enough that someone from that culture would say "yes, that's accurate" — avoid vague platitudes that could apply to any culture.
4. Cover DIFFERENT aspects of life — don't generate 5 norms all about dining. Spread across: greetings, food, hospitality, gifts, family, business, public behavior, celebrations, dress, communication, etc.
5. Each norm should be 10-25 words, written as a single clear sentence.

For each norm, also generate a matching GENERIC statement:
- The generic must share at least 2 key content words with the norm
- The generic must be a factual/technical/definitional statement (NOT about behavior)
- The generic must NOT use prescriptive language ("should", "must", "expected to")

Output ONLY in this exact format:
""" + "\n".join(f"[{i}] N: ...\n[{i}] G: ..." for i in range(1, num_to_generate + 1))


def parse_response(text, expected_count):
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
    for i in range(1, expected_count + 1):
        if i in pairs and "N" in pairs[i] and "G" in pairs[i]:
            results.append((pairs[i]["N"], pairs[i]["G"]))
        else:
            results.append(None)
    return results


def save_synthetic_pairs(pairs_to_save):
    with write_lock:
        file_exists = os.path.exists(SYNTHETIC_FILE)
        with open(SYNTHETIC_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["culture", "original_norm", "norm", "generic"])
            if not file_exists:
                writer.writeheader()
            for row in pairs_to_save:
                writer.writerow(row)


def log_progress(culture, status):
    with write_lock:
        with open(LOG_FILE, "a") as f:
            f.write(f"CULTURE={culture} | {status}\n")


def get_completed_cultures():
    if not os.path.exists(LOG_FILE):
        return set()
    completed = set()
    with open(LOG_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("CULTURE=") and "OK" in line:
                culture = line.split("=")[1].split("|")[0].strip()
                completed.add(culture)
    return completed


def generate_for_culture(client, culture, num_needed, existing_norms):
    """Generate synthetic norms for one culture using LLM world knowledge."""
    system_prompt = build_system_prompt()

    # Show which few-shot examples are being used
    print(f"  {culture}: using {len(existing_norms)} existing norms as few-shot examples")
    for i, norm in enumerate(existing_norms[:3]):
        print(f"    ex{i+1}: {norm[:80]}...")

    all_pairs = []
    remaining = num_needed
    while remaining > 0:
        batch_size = min(remaining, 20)
        user_prompt = build_generation_prompt(culture, batch_size, existing_norms)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,  # slightly higher for more diverse/creative norms
                )
                response_text = response.choices[0].message.content
                pairs = parse_response(response_text, batch_size)

                for pair in pairs:
                    if pair is not None:
                        all_pairs.append({
                            "culture": culture,
                            "original_norm": "[synthetic-knowledge]",
                            "norm": pair[0],
                            "generic": pair[1],
                        })

                remaining -= batch_size
                break

            except Exception as e:
                err = str(e)[:120]
                if "429" in err or "rate_limit" in err.lower():
                    wait = 10 * (attempt + 1)
                    print(f"  Rate limited for {culture}, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                else:
                    print(f"  ERROR for {culture}: {err}")
                    remaining -= batch_size
                    break

    if all_pairs:
        save_synthetic_pairs(all_pairs)

    return len(all_pairs)


def main():
    client = OpenAI(api_key=OPENAI_API_KEY)

    print(f"Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["culture", "norm", "generic"]).reset_index(drop=True)
    print(f"Loaded {len(df)} pairs, {df['culture'].nunique()} cultures")

    # Compute per-culture counts and target
    counts = df["culture"].value_counts()
    target = int(counts.quantile(0.75))  # 75th percentile — aggressive oversampling
    print(f"\nCulture count stats: min={counts.min()}, max={counts.max()}, median={counts.median():.0f}")
    print(f"Synthetic generation target (75th percentile): {target}")

    # Find cultures that need synthetic generation
    needs_synthetic = counts[counts < target]
    total_needed = sum(target - c for c in needs_synthetic)
    print(f"Cultures below median: {len(needs_synthetic)}")
    print(f"Total synthetic norms needed: {total_needed}")

    if total_needed == 0:
        print("No synthetic generation needed!")
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved unchanged to {OUTPUT_FILE}")
        return

    # Check which cultures are already done
    completed = get_completed_cultures()
    if completed:
        print(f"Already completed: {len(completed)} cultures (resuming)")

    # Remove synthetic file if starting fresh
    if not completed and os.path.exists(SYNTHETIC_FILE):
        os.remove(SYNTHETIC_FILE)

    # Build existing norms lookup (for few-shot examples)
    norms_by_culture = {}
    for culture in needs_synthetic.index:
        culture_norms = df[df["culture"] == culture]["norm"].tolist()
        norms_by_culture[culture] = culture_norms

    # Process cultures that need synthetic generation
    cultures_to_process = [
        (culture, target - count)
        for culture, count in needs_synthetic.items()
        if culture not in completed
    ]

    if not cultures_to_process:
        print("All cultures already processed!")
    else:
        est_calls = sum((n + 19) // 20 for _, n in cultures_to_process)
        print(f"\nCultures to process: {len(cultures_to_process)}")
        print(f"Estimated API calls: {est_calls}")
        print(f"Estimated time: ~{max(1, est_calls * 2 // NUM_WORKERS // 60)} min\n")

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for culture, num_needed in cultures_to_process:
                existing = norms_by_culture.get(culture, [])
                future = executor.submit(
                    generate_for_culture, client, culture, num_needed, existing
                )
                futures[future] = culture

            total_generated = 0
            for future in as_completed(futures):
                culture = futures[future]
                count = future.result()
                total_generated += count
                log_progress(culture, f"OK {count}")
                print(f"  {culture}: +{count} synthetic norms")

        print(f"\nGenerated {total_generated} synthetic pairs total")

    # Combine original + synthetic
    if os.path.exists(SYNTHETIC_FILE):
        df_synthetic = pd.read_csv(SYNTHETIC_FILE)
        df_combined = pd.concat([df, df_synthetic], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["norm"]).reset_index(drop=True)
    else:
        df_combined = df

    df_combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"SYNTHETIC GENERATION COMPLETE")
    print(f"  Original pairs: {len(df)}")
    print(f"  After synthetic: {len(df_combined)}")
    print(f"  Cultures: {df_combined['culture'].nunique()}")
    print(f"  Saved to: {OUTPUT_FILE}")

    # Show new distribution
    new_counts = df_combined["culture"].value_counts()
    print(f"\n  New min: {new_counts.min()} ({new_counts.idxmin()})")
    print(f"  New max: {new_counts.max()} ({new_counts.idxmax()})")
    print(f"  New median: {new_counts.median():.0f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
