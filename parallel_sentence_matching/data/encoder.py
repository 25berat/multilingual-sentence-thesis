#!/usr/bin/env python3


import math
from pathlib import Path
from typing import Dict, Any, Set

import pandas as pd

# ---- Paths: adjust if needed ----
INPUT_XLSX = Path(r"C:\Users\berat\OneDrive\Dokumente\Lang_Features_turkish.xlsx")
OUTPUT_XLSX = Path(r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\Lang_Features_encoded_turkish.xlsx")


# ---------- Helpers ----------

def _norm_col(name: str) -> str:
    """Normalize column name for matching."""
    s = str(name).strip().lower()
    return " ".join(s.split())


def _norm_cat(val: str) -> str:
    """Normalize category string for mapping."""
    s = str(val).strip().lower()
    return " ".join(s.split())


def parse_numeric_or_percent(x: Any) -> float:
    """
    Convert values to float where possible.

    Handles:
      - int/float
      - strings like "50", "12.3", "-4"
      - German-style "3,5", "270,0"
      - percentages "10%", "12,3%"  (we just return the numeric part)

    Returns float or NaN if not parseable.
    """
    if x is None:
        return math.nan

    if isinstance(x, float):
        if math.isnan(x):
            return math.nan
        return x

    if isinstance(x, int):
        return float(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return math.nan

        is_percent = s.endswith("%")
        if is_percent:
            s = s[:-1].strip()

        # Replace German decimal comma if there is no dot
        if "," in s and "." not in s:
            s = s.replace(",", ".")

        # Remove thin spaces / normal spaces as thousands separators
        s = s.replace("\u202f", "").replace(" ", "")

        try:
            val = float(s)
            return val
        except ValueError:
            return math.nan

    return math.nan


def encode_categorical_column(series: pd.Series, col_name: str) -> pd.Series:
    """
    Label-encode a categorical column:

    - Empty / 'none' / 'null' / 'n/a' / 'na' / NaN -> 0
    - Other unique values -> IDs 1..N (sorted by string)
    """
    cleaned = []
    for v in series:
        if isinstance(v, str):
            c = v.strip()
            if c == "" or c.lower() in {"none", "null", "n/a", "na"}:
                cleaned.append(None)
            else:
                cleaned.append(c)
        else:
            cleaned.append(None if pd.isna(v) else str(v).strip())

    # Unique categories (excluding None)
    unique_cats = sorted({c for c in cleaned if c is not None}, key=lambda x: str(x))
    cat2id: Dict[str, int] = {cat: i + 1 for i, cat in enumerate(unique_cats)}

    print(f"[ENCODE] Column '{col_name}': {len(unique_cats)} categories")
    # Mapping inspection (optional):
    # for cat, idx in cat2id.items():
    #     print(f"  {idx}: {cat}")
    # print("  0: <none/leer>")

    encoded_vals = [
        cat2id.get(c, 0) if c is not None else 0
        for c in cleaned
    ]
    return pd.Series(encoded_vals, index=series.index)


def encode_ordinal_column(series: pd.Series, col_name: str, mapping: Dict[str, int]) -> pd.Series:
    """
    Encode an ordinal column to numeric.

    - If a cell is already numeric/percentage -> keep numeric value.
    - Otherwise, try to map text by normalized category string.
    - Unknown categories -> NaN.
    """
    encoded = []
    for v in series:
        # First try numeric
        num = parse_numeric_or_percent(v)
        if not (isinstance(num, float) and math.isnan(num)):
            encoded.append(num)
            continue

        # Then try text mapping
        if isinstance(v, str):
            key = _norm_cat(v)
            if key in mapping:
                encoded.append(mapping[key])
            else:
                print(f"[WARN] Ordinal column '{col_name}': unknown category '{v}'")
                encoded.append(math.nan)
        else:
            encoded.append(math.nan)

    print(f"[ORDINAL] Column '{col_name}' encoded with mapping {mapping}")
    return pd.Series(encoded, index=series.index)


# ---------- Configuration ----------

# Columns we do NOT touch (keep original as-is, no encoding)
DO_NOT_ENCODE_NORM: Set[str] = {
    _norm_col("lang"),
    _norm_col("language"),
    _norm_col("iso639_3"),
}

# Columns that should be treated as numeric
NUMERIC_COLS_NORM: Set[str] = {
    # from first step
    _norm_col("Latitude"),
    _norm_col("Longitude"),
    _norm_col("Distance_geo_tur"),
    _norm_col("official language (no. Of states)"),
    _norm_col("speakers"),
    _norm_col("main script"),
    _norm_col("script changed"),

    # new numeric-ish columns (no bracket in your spec)
    _norm_col("Alphabet size"),
    _norm_col("wiki_size_bucket"),
    _norm_col("is_training_lang_glot500"),
    _norm_col("is_training_lang_labse"),
    _norm_col("is_training_lang_laser"),
    _norm_col("is_training_lang_sonar"),
    _norm_col("is_training_lang_xlmr"),
    _norm_col("dist_deu_avg_distance"),
    _norm_col("dist_deu_glot500"),
    _norm_col("dist_deu_labse"),
    _norm_col("dist_deu_xlmr"),
    _norm_col("dist_deu_laser"),
    _norm_col("dist_deu_sonar"),
    _norm_col("linguisticdiversity"),
    _norm_col("hasGender"),
    _norm_col("Language Power Index"),
    _norm_col("writing direction"),
    _norm_col("TTR_glot500"),
    _norm_col("TTR_labse"),
    _norm_col("TTR_xlmr"),
    _norm_col("TTR_whitspace")
}

# Categorical text columns (label-encoded -> <col>_enc)
CATEGORICAL_COLS_NORM: Set[str] = {
    # from first step
    _norm_col("macroarea"),
    _norm_col("script"),
    _norm_col("family"),
    _norm_col("subfamily"),
    _norm_col("colonizer"),
    # new text features
    _norm_col("Order of Genitive and Noun"),
    _norm_col("Order of Adjective and Noun"),
    _norm_col("Order of Relative Clause and Noun"),
    _norm_col("Order of Negative Morpheme and Verb"),
    _norm_col("Position of Interrogative Phrases in Content Questions"),
    _norm_col("Writing Systems"),
    _norm_col("Coding of Nominal Plurality"),
    _norm_col("Definite Articles"),
    _norm_col("NumberOfCases"),
    _norm_col("Expression of Pronominal Subjects"),
    _norm_col("Prefixing vs. Suffixing in Inflectional Morphology"),
    _norm_col("Locus of Marking: Whole-language Typology"),
    _norm_col("Order of Adposition and Noun Phrase"),
    _norm_col("word_order"),

}

# Ordinal columns (text -> numeric codes in <col>_num)
ORDINAL_COL_MAPPINGS_NORM: Dict[str, Dict[str, int]] = {
    _norm_col("Inflectional Synthesis of the Verb"): {
        _norm_cat("low"): 0,
        _norm_cat("medium"): 1,
        _norm_cat("high"): 2,
    },
    _norm_col("Official status"): {
        _norm_cat("No"): 0,
        _norm_cat("regional"): 1,
        _norm_cat("official"): 2,
    },
    _norm_col("has_Tone"): {
        _norm_cat("No tone"): 0,
        _norm_cat("Simple"): 1,
        _norm_cat("Complex"): 2,
    },
    _norm_col("Consonant Inventories"): {
        # you defined: Moderately small=0, Average=1, Moderately large=2, Large=3
        # If "Small" appears, we map it also to 0.
        _norm_cat("Small"): 0,
        _norm_cat("Moderately small"): 0,
        _norm_cat("Average"): 1,
        _norm_cat("Moderately large"): 2,
        _norm_cat("Large"): 3,
    },
    _norm_col("Vowel Quality Inventories"): {
        # you defined: Moderately small=0, Average=1, Moderately large=2, Large=3
        # If "Small" appears, we map it also to 0.
        _norm_cat("Small"): 0,
        _norm_cat("Average"): 1,
        _norm_cat("Large"): 2,
    },

    _norm_col("Syllable Structure"): {
        # you defined: small=0, average=1, large=2
        # map WALS-style labels to these buckets
        _norm_cat("Moderately complex"): 0,
        _norm_cat("Complex"): 1,
    },
    _norm_col("Number of Genders"): {
        _norm_cat("No"): 0,
        _norm_cat("Two"): 1,
        _norm_cat("Three"): 2,
        _norm_cat("Four"): 3,
        _norm_cat("Five or more"): 3,
    },
    _norm_col("hasCases"): {
        _norm_cat("No"): 0,
        _norm_cat("none"): 0,
        _norm_cat("yes"): 1,
    },
    _norm_col("Numeral Classifiers"): {
        _norm_cat("Absent"): 0,
        _norm_cat("Optional"): 1,
        _norm_cat("Obligatory"): 2,
    },
    _norm_col("Reduplication"): {
        _norm_cat("No reduplication"): 0,
        _norm_cat("No productive reduplication"): 0,
        _norm_cat("Full only"): 1,
        _norm_cat("Full reduplication only"): 1,
        _norm_cat("Full and partial"): 2,
        _norm_cat("Productive full and partial reduplication"): 2,
    },
}


# ---------- Main ----------

def main():
    print(f"Reading: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX)

    # Map normalized name -> actual name in df
    norm_to_real = {_norm_col(c): c for c in df.columns}

    for col in df.columns:
        norm = _norm_col(col)

        if norm in DO_NOT_ENCODE_NORM:
            print(f"[SKIP] '{col}' (do not encode)")
            continue

        # Ordinal columns: create <col>_num
        if norm in ORDINAL_COL_MAPPINGS_NORM:
            enc_col_name = f"{col}_num"
            if enc_col_name in df.columns:
                print(f"[WARN] Ordinal numeric column '{enc_col_name}' already exists, it will be overwritten.")
            mapping = ORDINAL_COL_MAPPINGS_NORM[norm]
            df[enc_col_name] = encode_ordinal_column(df[col], col, mapping)
            # Keep original text column as well
            continue

        # Pure numeric columns: parse to float in-place
        if norm in NUMERIC_COLS_NORM:
            print(f"[NUMERIC] Normalizing column '{col}'")
            df[col] = df[col].apply(parse_numeric_or_percent)
            continue

        # Categorical columns: create <col>_enc
        if norm in CATEGORICAL_COLS_NORM:
            enc_col_name = f"{col}_enc"
            if enc_col_name in df.columns:
                print(f"[WARN] Encoded column '{enc_col_name}' already exists, it will be overwritten.")
            df[enc_col_name] = encode_categorical_column(df[col], col)
            continue

        # For now, any other columns (not listed yet) are left untouched.
        print(f"[INFO] Leaving column '{col}' unchanged (no rules yet).")

    print(f"Saving encoded file to: {OUTPUT_XLSX}")
    df.to_excel(OUTPUT_XLSX, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
