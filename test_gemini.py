"""
Quick test: Generate 10 factual (non-norm) sentences using Gemini
to see if the approach works before full-scale generation.
"""

import os
from google import genai

# ── CONFIG ──
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# ── SAMPLE NORMS (to show Gemini what norms look like, so it avoids them) ──
SAMPLE_NORMS = [
    ("American", "tip as an expression of gratitude and appreciation for good service"),
    ("Japanese", "remove their shoes before entering homes"),
    ("Italian",  "drink, offer, and make grappa, a traditional Italian alcoholic beverage"),
    ("German",   "follow strict recycling rules and separate waste into multiple bins"),
    ("Korean",   "pour drinks for elders using both hands as a sign of respect"),
]

# ── PROMPT ──
prompt = f"""Here are examples of CULTURAL NORMS (behavioral expectations of a group):
{chr(10).join(f'  - [{country}] {norm}' for country, norm in SAMPLE_NORMS)}

Now generate exactly 10 FACTUAL statements (NOT norms) about these countries.
Rules:
1. Generate 2 facts each for: United States, Japan, Italy, Germany, South Korea
2. Facts must be verifiable — geography, history, economy, demographics, science, etc.
3. Do NOT include customs, traditions, behavioral expectations, or what people typically do.
4. Each fact must be 8-22 words long (similar length to the norms above).
5. Vary sentence structure — don't start every fact with the country name.
6. Output format: one fact per line, prefixed with the country in brackets like [United States]

BAD (these are norms, not facts):
  - [Japanese] People bow when greeting each other
  - [German] People tend to be very punctual

GOOD (these are facts):
  - [Japan] Mount Fuji stands at 3,776 meters as the country's tallest peak
  - [Germany] The Rhine River flows 1,230 kilometers through six European countries

Output ONLY the 10 facts, nothing else:"""

print("Sending prompt to Gemini...\n")

# Try multiple models in case one has quota
for model_name in ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]:
    try:
        print(f"Trying model: {model_name}")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        print(f"Success with {model_name}!\n")
        break
    except Exception as e:
        print(f"  Failed: {e}\n")
        continue
else:
    print("All models failed. Check your API key quota.")
    exit(1)

print("=== GEMINI RESPONSE ===")
print(response.text)

# ── PARSE & SAVE ──
import csv
import re

OUTPUT_FILE = "test_gemini_output.csv"

lines = [l.strip() for l in response.text.strip().split("\n") if l.strip()]
rows = []
for line in lines:
    # Parse "[Country] fact text"
    match = re.match(r"\[(.+?)\]\s*(.+)", line)
    if match:
        country = match.group(1).strip()
        text = match.group(2).strip()
        word_count = len(text.split())
        rows.append({"country": country, "text": text, "word_count": word_count})

# Combine norms + facts into one CSV
country_aliases = {
    "American": "United States", "Japanese": "Japan",
    "Italian": "Italy", "German": "Germany", "Korean": "South Korea"
}
all_rows = []
for country, norm in SAMPLE_NORMS:
    all_rows.append({
        "country": country_aliases.get(country, country),
        "text": norm,
        "label": "norm",
        "word_count": len(norm.split())
    })
for row in rows:
    all_rows.append({
        "country": row["country"],
        "text": row["text"],
        "label": "fact",
        "word_count": row["word_count"]
    })

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["country", "label", "text", "word_count"])
    writer.writeheader()
    writer.writerows(all_rows)

print(f"\nSaved to {OUTPUT_FILE}")

# ── PRINT BOTH SIDE BY SIDE ──
print("\n" + "="*80)
print("COMPARISON: ORIGINAL NORMS vs GENERATED FACTS")
print("="*80)

# Map sample norms by country
norm_map = {}
for country, norm in SAMPLE_NORMS:
    norm_map.setdefault(country, []).append(norm)

# Map generated facts by country
fact_map = {}
for row in rows:
    fact_map.setdefault(row["country"], []).append(row["text"])

# Print comparison
all_countries = list(dict.fromkeys(
    [c for c, _ in SAMPLE_NORMS] + [r["country"] for r in rows]
))
for country in all_countries:
    print(f"\n--- {country} ---")
    for norm in norm_map.get(country, []):
        print(f"  [NORM]  {norm}")
    # Match country names (American -> United States, etc.)
    country_aliases = {
        "American": "United States", "Japanese": "Japan",
        "Italian": "Italy", "German": "Germany", "Korean": "South Korea"
    }
    fact_key = country_aliases.get(country, country)
    for fact in fact_map.get(fact_key, []):
        print(f"  [FACT]  {fact}")

print("\n" + "="*80)
