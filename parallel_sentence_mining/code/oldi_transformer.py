from pathlib import Path
import re
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
IN_DIR = Path(r"C:\Users\berat\PycharmProjects\Belopsem\raw_data\monolingual_data")
GLOB_PATTERN = "*.parquet"   # use "**/*.parquet" for recursive
TEXT_COL = "text"

# the special German txt file
DE_TXT_IN = IN_DIR / "deu_wikipedia_2021_10K-sentences.txt"
DE_TXT_OUT = IN_DIR / "deu_Latn.txt"

# matches: "123 ; text" | "123\ttext" | "123: text" | "123) text" | "123. text" | "123 - text"
LEADING_NUM_RE = re.compile(r"^\s*\d+\s*(?:[;:)\].\-–—]|,\s*|/|\||\t)\s*")

# ============================================================
# PARQUET: keep only `text` column and overwrite parquet
# ============================================================
def clean_line(s: str) -> str:
    s = s.replace("\r", " ").replace("\n", " ")
    return " ".join(s.split()).strip()

def process_parquet(fp: Path) -> None:
    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        print(f"[SKIP] Failed to read: {fp.name} -> {e}")
        return

    if TEXT_COL not in df.columns:
        print(f"[SKIP] No '{TEXT_COL}' column: {fp.name} (cols={list(df.columns)})")
        return

    text = df[TEXT_COL].dropna().astype(str).map(clean_line)
    text = text[text != ""]

    out_df = pd.DataFrame({TEXT_COL: text})

    try:
        out_df.to_parquet(fp, index=False)  # overwrite same file
        print(f"[OK] Overwrote {fp.name} with text-only ({len(out_df)} rows)")
    except Exception as e:
        print(f"[SKIP] Failed to write: {fp.name} -> {e}")

# ============================================================
# TXT: clean numbering ONLY for that one file, then rename to deu_Latn.txt
# ============================================================
def clean_and_rename_de_txt() -> None:
    if not DE_TXT_IN.exists():
        print(f"[INFO] TXT not found, skipping: {DE_TXT_IN}")
        return

    lines = DE_TXT_IN.read_text(encoding="utf-8", errors="replace").splitlines()

    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line = LEADING_NUM_RE.sub("", line).strip()
        if line:
            cleaned.append(line)

    # write cleaned content to new filename
    DE_TXT_OUT.write_text("\n".join(cleaned), encoding="utf-8")
    print(f"[OK] Wrote cleaned German txt: {DE_TXT_OUT.name} ({len(cleaned)} lines)")

    # optionally delete old file (so you don't keep duplicates)
    try:
        DE_TXT_IN.unlink()
        print(f"[OK] Deleted old file: {DE_TXT_IN.name}")
    except Exception as e:
        print(f"[WARN] Could not delete old file {DE_TXT_IN.name}: {e}")

# ============================================================
# MAIN
# ============================================================
def main():
    # 1) process parquet files
    files = sorted(IN_DIR.glob(GLOB_PATTERN))
    if not files:
        print(f"[INFO] No parquet files found in {IN_DIR} with pattern {GLOB_PATTERN}")
    else:
        for fp in files:
            process_parquet(fp)

    # 2) fix the one German txt file + rename
    clean_and_rename_de_txt()

    print("\nDone.")

if __name__ == "__main__":
    main()
