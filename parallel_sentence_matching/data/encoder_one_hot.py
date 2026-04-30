#!/usr/bin/env python3

import math
from pathlib import Path
from typing import Any, Set, List, Dict

import numpy as np
import pandas as pd

# ---- Paths: adjust if needed ----
INPUT_XLSX = Path(r"C:\Users\berat\OneDrive\Dokumente\Lang_Features_turkish.xlsx")
OUTPUT_XLSX = Path(r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\Lang_Features_encoded_onehot_turkish.xlsx")
DROPPED_REPORT_CSV = OUTPUT_XLSX.with_suffix("").with_name(OUTPUT_XLSX.stem + "_dropped_categories.csv")

MIN_CATEGORY_COUNT = 4  # drop categories with n < 4


# ---------- Helpers ----------

def _norm_col(name: str) -> str:
    s = str(name).strip().lower()
    return " ".join(s.split())


def parse_numeric_or_percent(x: Any) -> float:
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

        if "," in s and "." not in s:
            s = s.replace(",", ".")

        s = s.replace("\u202f", "").replace(" ", "")

        try:
            return float(s)
        except ValueError:
            return math.nan

    return math.nan


def _clean_cat_value(v: Any) -> Any:
    """Normalize a single category value; return np.nan for empty/NA-like."""
    if isinstance(v, str):
        c = v.strip()
        if c == "" or c.lower() in {"none", "null", "n/a", "na"}:
            return np.nan
        return c
    if pd.isna(v):
        return np.nan
    return str(v).strip()


def one_hot_encode_column(
    series: pd.Series,
    col_name: str,
    dropped_report: List[Dict[str, Any]],
    min_count: int = MIN_CATEGORY_COUNT,
) -> pd.DataFrame:
    """
    One-hot encode a categorical column, but:
      - compute category frequencies
      - drop categories with n < min_count
      - rename dummy columns to include (n=...)
    """
    tmp = series.apply(_clean_cat_value).astype("object")

    # counts excluding NaN
    counts = tmp.value_counts(dropna=True)

    # decide what to keep/drop
    keep_cats = counts[counts >= min_count].index.tolist()
    drop_cats = counts[counts < min_count]

    for cat, n in drop_cats.items():
        dropped_report.append({
            "column": col_name,
            "category": cat,
            "n": int(n),
            "reason": f"n<{min_count}",
        })

    # if nothing survives, return empty df (same index)
    if len(keep_cats) == 0:
        print(f"[ONE-HOT] Column '{col_name}': 0 kept (all categories < {min_count})")
        return pd.DataFrame(index=series.index)

    # restrict series to kept categories (others -> NaN)
    tmp_kept = tmp.where(tmp.isin(keep_cats), other=np.nan)

    dummies = pd.get_dummies(
        tmp_kept,
        prefix=col_name,
        prefix_sep="__",
        dummy_na=False,
        dtype=int,
    ).astype(int)

    # rename columns to include (n=...)
    rename_map = {}
    for dummy_col in dummies.columns:
        # expected: f"{col_name}__{category}"
        prefix = f"{col_name}__"
        if dummy_col.startswith(prefix):
            cat = dummy_col[len(prefix):]
            n = int(counts.get(cat, 0))
            rename_map[dummy_col] = f"{col_name}__{cat} (n={n})"

    dummies = dummies.rename(columns=rename_map)

    print(f"[ONE-HOT] Column '{col_name}': kept {len(keep_cats)} / total {len(counts)} categories "
          f"(dropped {len(drop_cats)}) -> {dummies.shape[1]} dummy columns")
    return dummies


# ---------- Configuration ----------

DO_NOT_ENCODE_NORM: Set[str] = {
    _norm_col("lang"),
    _norm_col("language"),
    _norm_col("iso639_3"),
}

NUMERIC_COLS_NORM: Set[str] = {
    _norm_col("Latitude"),
    _norm_col("Longitude"),
    _norm_col("Distance_geo_tur"),
    _norm_col("official language (no. Of states)"),
    _norm_col("speakers"),
    _norm_col("main script"),
    _norm_col("script changed"),

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
    _norm_col("dist_deu_laser"),
    _norm_col("dist_deu_sonar"),
    _norm_col("dist_deu_qwen3"),
    _norm_col("dist_deu_llama31"),
    _norm_col("linguisticdiversity"),
    _norm_col("hasGender"),
    _norm_col("Language Power Index"),
    _norm_col("writing direction"),
    _norm_col("TTR_glot500"),
    _norm_col("TTR_labse"),
    _norm_col("TTR_xlmr"),
    _norm_col("TTR_whitspace"),
    _norm_col("glot500_size")
}

CATEGORICAL_COLS_NORM: Set[str] = {
    _norm_col("macroarea"),
    _norm_col("script"),
    _norm_col("family"),
    _norm_col("subfamily"),
    _norm_col("colonizer"),

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
    _norm_col("Inflectional Synthesis of the Verb"),
    _norm_col("Official status"),
    _norm_col("has_Tone"),
    _norm_col("Consonant Inventories"),
    _norm_col("Vowel Quality Inventories"),
    _norm_col("Syllable Structure"),
    _norm_col("Number of Genders"),
    _norm_col("hasCases"),
    _norm_col("Numeral Classifiers"),
    _norm_col("Reduplication"),
}


# ---------- Main ----------

def main():
    print(f"Reading: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX)

    dummy_blocks = []
    dropped_report: List[Dict[str, Any]] = []

    for col in list(df.columns):
        norm = _norm_col(col)

        if norm in DO_NOT_ENCODE_NORM:
            print(f"[SKIP] '{col}' (do not encode)")
            continue

        if norm in NUMERIC_COLS_NORM:
            print(f"[NUMERIC] Normalizing column '{col}'")
            df[col] = df[col].apply(parse_numeric_or_percent)
            continue

        if norm in CATEGORICAL_COLS_NORM:
            dummies = one_hot_encode_column(
                df[col],
                col,
                dropped_report=dropped_report,
                min_count=MIN_CATEGORY_COUNT,
            )
            if not dummies.empty:
                dummy_blocks.append(dummies)
            continue

        print(f"[INFO] Leaving column '{col}' unchanged (no rules yet).")

    if dummy_blocks:
        print(f"[INFO] Concatenating {len(dummy_blocks)} dummy blocks.")
        df = pd.concat([df] + dummy_blocks, axis=1)

    print(f"Saving encoded file to: {OUTPUT_XLSX}")
    df.to_excel(OUTPUT_XLSX, index=False)

    # save dropped report
    if dropped_report:
        rep_df = pd.DataFrame(dropped_report).sort_values(["column", "n", "category"], ascending=[True, True, True])
        rep_df.to_csv(DROPPED_REPORT_CSV, index=False, encoding="utf-8")
        print(f"[REPORT] Dropped categories saved to: {DROPPED_REPORT_CSV} ({len(rep_df)} rows)")
    else:
        print("[REPORT] No categories were dropped.")

    print("Done.")


if __name__ == "__main__":
    main()
