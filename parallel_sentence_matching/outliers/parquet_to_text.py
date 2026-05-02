# parquet_to_text.py
"""
Scan SOURCE_DIR for *.parquet (e.g., deu_Latn.parquet),
extract the 'text' column, and write one-sentence-per-line .txt files
into TARGET_DIR with the same base name.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Iterable, Tuple
import pandas as pd

# === Paths ===
SOURCE_DIR = Path(r"C:\Users\berat\PycharmProjects\outliers\data\flores_raw_data")
TARGET_DIR = Path(r"C:\Users\berat\PycharmProjects\outliers\data\bucc2018\2")

# === Settings ===
TEXT_COL = "text"          # change if your column name differs
DEDUP = True
MIN_LEN = 1
MAX_LEN = 0                # 0 = no limit
STRIP_QUOTES = True

# Optional hard filters (set to None to disable)
FILTER_ISO_639_3: Optional[str] = None   # e.g. "deu"
FILTER_ISO_15924: Optional[str] = None   # e.g. "Latn"
FILTER_GLOTTOCODE: Optional[str] = None  # e.g. "stan1295"

# Infer iso/script from filename like deu_Latn.parquet
INFER_FILTERS_FROM_FILENAME = True


def normalize_text(s: str, strip_quotes: bool = False) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if strip_quotes:
        repl = {
            "\u201C": '"', "\u201D": '"', "\u00AB": '"', "\u00BB": '"',
            "\u2018": "'", "\u2019": "'", "\u2013": "-", "\u2014": "-",
        }
        for k, v in repl.items():
            s = s.replace(k, v)
    return s


def maybe_infer_filters_from_name(parquet_path: Path) -> tuple[Optional[str], Optional[str]]:
    stem = parquet_path.stem  # e.g., "deu_Latn"
    m = re.match(r"^([a-z]{3})[_\-]([A-Za-z]{4})$", stem)
    if m:
        return m.group(1), m.group(2)
    m2 = re.match(r"^([a-z]{3})[_\-]([A-Za-z]{4})[_\-].*$", stem)
    if m2:
        return m2.group(1), m2.group(2)
    return None, None


def apply_filters(df: pd.DataFrame,
                  iso_639_3: Optional[str],
                  iso_15924: Optional[str],
                  glottocode: Optional[str]) -> pd.DataFrame:
    for col, val in (("iso_639_3", iso_639_3),
                     ("iso_15924", iso_15924),
                     ("glottocode", glottocode)):
        if val is not None and col in df.columns:
            df = df[df[col] == val]
    return df


def parquet_to_txt(inp: Path, out: Path,
                   text_col: str = TEXT_COL,
                   iso_639_3: Optional[str] = FILTER_ISO_639_3,
                   iso_15924: Optional[str] = FILTER_ISO_15924,
                   glottocode: Optional[str] = FILTER_GLOTTOCODE,
                   dedup: bool = DEDUP,
                   min_len: int = MIN_LEN,
                   max_len: int = MAX_LEN,
                   strip_quotes: bool = STRIP_QUOTES) -> int:
    df = pd.read_parquet(inp)  # if this errors, install pyarrow: pip install pyarrow

    if INFER_FILTERS_FROM_FILENAME and (iso_639_3 is None or iso_15924 is None):
        f_iso, f_scr = maybe_infer_filters_from_name(inp)
        iso_639_3 = iso_639_3 or f_iso
        iso_15924 = iso_15924 or f_scr

    df = apply_filters(df, iso_639_3, iso_15924, glottocode)

    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {inp.name}. Available: {list(df.columns)}")

    s = df[text_col].astype(str).map(lambda x: normalize_text(x, strip_quotes=strip_quotes))

    if min_len > 0:
        s = s[s.str.len() >= min_len]
    if max_len and max_len > 0:
        s = s[s.str.len() <= max_len]
    if dedup:
        s = s.drop_duplicates()

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for line in s:
            f.write(line + "\n")
    return int(s.shape[0])


def find_parquets(directory: Path) -> Iterable[Path]:
    return sorted(directory.glob("*.parquet"))


def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    parquets = list(find_parquets(SOURCE_DIR))
    if not parquets:
        print(f"[INFO] No .parquet files in {SOURCE_DIR}")
        return

    print(f"[INFO] Found {len(parquets)} parquet file(s) in {SOURCE_DIR}")
    total = 0
    for p in parquets:
        out = TARGET_DIR / (p.stem + ".txt")
        try:
            n = parquet_to_txt(p, out)
            total += n
            print(f"[OK] {p.name} -> {out.name} ({n} lines)")
        except Exception as e:
            print(f"[ERROR] {p.name}: {e}")
    print(f"[DONE] Total lines written: {total} → {TARGET_DIR}")


if __name__ == "__main__":
    main()
