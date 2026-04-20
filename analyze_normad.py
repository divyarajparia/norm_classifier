"""
Analyze and clean the NormAd dataset (akhilayerukola/NormAd).
- Loads from HuggingFace
- Prints distributions (country, gold label, subaxis)
- Normalizes country names to demonyms (to match CultureBank style)
- Keeps ALL 2,633 rows (Rule-of-Thumb is a valid norm for all gold labels)
- Saves normad_cleaned.csv

Output columns: culture, norm_text, subaxis, value
"""

import pandas as pd
from datasets import load_dataset

OUTPUT_CSV = "normad_cleaned.csv"

# Country name → demonym mapping (75 countries)
COUNTRY_TO_DEMONYM = {
    "afghanistan": "Afghan",
    "argentina": "Argentine",
    "australia": "Australian",
    "austria": "Austrian",
    "bangladesh": "Bangladeshi",
    "bosnia_and_herzegovina": "Bosnian",
    "brazil": "Brazilian",
    "cambodia": "Cambodian",
    "canada": "Canadian",
    "chile": "Chilean",
    "china": "Chinese",
    "colombia": "Colombian",
    "croatia": "Croatian",
    "cyprus": "Cypriot",
    "egypt": "Egyptian",
    "ethiopia": "Ethiopian",
    "fiji": "Fijian",
    "france": "French",
    "germany": "German",
    "greece": "Greek",
    "hong_kong": "Hong Konger",
    "hungary": "Hungarian",
    "india": "Indian",
    "indonesia": "Indonesian",
    "iran": "Iranian",
    "iraq": "Iraqi",
    "ireland": "Irish",
    "israel": "Israeli",
    "italy": "Italian",
    "japan": "Japanese",
    "kenya": "Kenyan",
    "laos": "Laotian",
    "lebanon": "Lebanese",
    "malaysia": "Malaysian",
    "malta": "Maltese",
    "mauritius": "Mauritian",
    "mexico": "Mexican",
    "myanmar": "Burmese",
    "nepal": "Nepalese",
    "netherlands": "Dutch",
    "new_zealand": "New Zealander",
    "north_macedonia": "Macedonian",
    "pakistan": "Pakistani",
    "palestinian_territories": "Palestinian",
    "papua_new_guinea": "Papua New Guinean",
    "peru": "Peruvian",
    "philippines": "Filipino",
    "poland": "Polish",
    "portugal": "Portuguese",
    "romania": "Romanian",
    "russia": "Russian",
    "samoa": "Samoan",
    "saudi_arabia": "Saudi Arabian",
    "serbia": "Serbian",
    "singapore": "Singaporean",
    "somalia": "Somali",
    "south_africa": "South African",
    "south_korea": "South Korean",
    "south_sudan": "South Sudanese",
    "spain": "Spanish",
    "sri_lanka": "Sri Lankan",
    "sudan": "Sudanese",
    "sweden": "Swedish",
    "syria": "Syrian",
    "taiwan": "Taiwanese",
    "thailand": "Thai",
    "timor-leste": "Timorese",
    "tonga": "Tongan",
    "türkiye": "Turkish",
    "ukraine": "Ukrainian",
    "united_kingdom": "British",
    "united_states_of_america": "American",
    "venezuela": "Venezuelan",
    "vietnam": "Vietnamese",
    "zimbabwe": "Zimbabwean",
}


def main():
    print("Loading NormAd dataset from HuggingFace...")
    ds = load_dataset("akhilayerukola/NormAd")
    df = pd.DataFrame(ds["train"])

    print(f"\nTotal rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # ── Gold Label Distribution ──
    print("\n=== GOLD LABEL DISTRIBUTION ===")
    print(df["Gold Label"].value_counts().to_string())

    # ── Country Distribution ──
    print(f"\n=== COUNTRY DISTRIBUTION ({df['Country'].nunique()} countries) ===")
    vc = df["Country"].value_counts()
    print(f"Min: {vc.min()} ({vc.idxmin()}) | Max: {vc.max()} ({vc.idxmax()}) | Median: {vc.median():.0f}")
    print(vc.to_string())

    # ── Subaxis Distribution ──
    print(f"\n=== SUBAXIS DISTRIBUTION ({df['Subaxis'].nunique()} types) ===")
    print(df["Subaxis"].value_counts().to_string())

    # ── Word lengths ──
    print("\n=== WORD LENGTHS ===")
    for col in ["Value", "Rule-of-Thumb", "Story"]:
        wc = df[col].str.split().str.len()
        print(f"  {col}: mean={wc.mean():.1f}, min={wc.min()}, max={wc.max()}")

    # ── Normalize country names to demonyms ──
    print("\n=== NORMALIZING COUNTRIES TO DEMONYMS ===")
    unmapped = set(df["Country"].unique()) - set(COUNTRY_TO_DEMONYM.keys())
    if unmapped:
        print(f"WARNING: Unmapped countries: {unmapped}")

    df["culture"] = df["Country"].map(COUNTRY_TO_DEMONYM)
    unmapped_rows = df["culture"].isna().sum()
    if unmapped_rows > 0:
        print(f"WARNING: {unmapped_rows} rows with unmapped countries, keeping original name")
        df.loc[df["culture"].isna(), "culture"] = df.loc[df["culture"].isna(), "Country"]

    # ── Build cleaned output ──
    df_clean = pd.DataFrame({
        "culture": df["culture"],
        "norm_text": df["Rule-of-Thumb"].str.strip(),
        "subaxis": df["Subaxis"].str.strip(),
        "value": df["Value"].str.strip(),
    })

    df_clean = df_clean.dropna(subset=["norm_text"]).reset_index(drop=True)
    df_clean = df_clean[df_clean["norm_text"] != ""].reset_index(drop=True)

    print(f"\n=== CLEANED DATA ===")
    print(f"Rows: {len(df_clean)}")
    print(f"Unique cultures: {df_clean['culture'].nunique()}")
    print(f"\nCulture distribution (demonyms):")
    print(df_clean["culture"].value_counts().to_string())

    # ── Check overlap with CultureBank ──
    try:
        cb = pd.read_csv("generated_pairs_openai.csv")
        cb_cultures = set(cb["culture"].str.strip().unique())
        normad_cultures = set(df_clean["culture"].unique())
        overlap = cb_cultures & normad_cultures
        cb_only = cb_cultures - normad_cultures
        normad_only = normad_cultures - cb_cultures

        print(f"\n=== OVERLAP WITH CULTUREBANK ===")
        print(f"CultureBank cultures: {len(cb_cultures)}")
        print(f"NormAd cultures: {len(normad_cultures)}")
        print(f"Overlap: {len(overlap)}")
        if overlap:
            print(f"  Shared: {sorted(overlap)}")
        print(f"NormAd only ({len(normad_only)}): {sorted(normad_only)[:20]}{'...' if len(normad_only)>20 else ''}")
        print(f"CultureBank only ({len(cb_only)}): {sorted(cb_only)[:20]}{'...' if len(cb_only)>20 else ''}")
    except FileNotFoundError:
        print("\ngenerated_pairs_openai.csv not found, skipping overlap analysis")

    # ── Save ──
    df_clean.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")

    # ── Sample rows ──
    print("\n=== SAMPLE ROWS ===")
    for i in [0, 100, 500, 1000, 2000]:
        if i < len(df_clean):
            row = df_clean.iloc[i]
            print(f"\n[{i}] {row['culture']} ({row['subaxis']})")
            print(f"  Norm: {row['norm_text']}")
            print(f"  Value: {row['value']}")


if __name__ == "__main__":
    main()
