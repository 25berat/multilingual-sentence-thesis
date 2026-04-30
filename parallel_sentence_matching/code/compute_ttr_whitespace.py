# compute_ttr_whitespace.py
# Adds: a sorted TABLE PNG (like your screenshot)

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt


INPUT_DIR = Path(r"/data2/unshuffled_custom_data")
OUTPUT_DIR = Path(r"C:\Users\berat\PycharmProjects\PaSeMiLL\results\TTR\whitespace")

OUTPUT_CSV = OUTPUT_DIR / "ttr_whitespace_normalized.csv"
OUTPUT_BAR_PNG = OUTPUT_DIR / "ttr_whitespace_sorted.png"
OUTPUT_TABLE_PNG = OUTPUT_DIR / "ttr_whitespace_sorted_table.png"


def normalize_token(tok: str) -> str:
    """
    Split by whitespace, but normalize so 'pilot' and 'pilot.' are NOT different:
    - remove Unicode punctuation (P*) and symbols (S*)
    - casefold
    """
    tok = tok.strip()
    if not tok:
        return ""

    chars = []
    for ch in tok:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            continue
        chars.append(ch)

    return "".join(chars).casefold().strip()


def compute_ttr_for_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_tokens = text.split()  # whitespace split
    tokens = [normalize_token(t) for t in raw_tokens]
    tokens = [t for t in tokens if t]

    num_tokens = len(tokens)
    num_types = len(set(tokens))
    ttr = (num_types / num_tokens) if num_tokens > 0 else 0.0

    return {
        "file": path.name,
        "language": path.stem,
        "tokens": num_tokens,
        "types": num_types,
        "ttr": ttr,
    }


def save_csv(rows: list[dict]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "language", "tokens", "types", "ttr"])
        writer.writeheader()
        writer.writerows(rows)


def save_sorted_bar_png(rows: list[dict]) -> None:
    rows_sorted = sorted(rows, key=lambda r: r["ttr"], reverse=True)
    languages = [r["language"] for r in rows_sorted]
    ttrs = [r["ttr"] for r in rows_sorted]

    fig_w = max(12, len(languages) * 0.30)
    plt.figure(figsize=(fig_w, 6))
    plt.bar(languages, ttrs)
    plt.xticks(rotation=90)
    plt.ylabel("TTR")
    plt.title("Whitespace-normalized TTR per language (sorted high → low)")
    plt.tight_layout()
    plt.savefig(OUTPUT_BAR_PNG, dpi=200)
    plt.close()


def save_sorted_table_png(rows: list[dict], max_rows: int | None = None) -> None:
    """
    Creates a PNG table sorted by TTR (high -> low), like your screenshot.
    If max_rows is set (e.g., 40), it will only render the top N rows to keep it readable.
    """
    rows_sorted = sorted(rows, key=lambda r: r["ttr"], reverse=True)
    if max_rows is not None:
        rows_sorted = rows_sorted[:max_rows]

    headers = ["language", "tokens", "types", "ttr"]

    # format numbers similar to your screenshot (commas + fixed decimals)
    cell_text = []
    for r in rows_sorted:
        cell_text.append([
            r["language"],
            f'{r["tokens"]:,}',
            f'{r["types"]:,}',
            f'{r["ttr"]:.6f}',
        ])

    # figure size: scale height with number of rows
    n = len(cell_text)
    fig_h = max(6, 0.28 * (n + 1))   # +1 header row
    fig_w = 8                        # adjust if you want wider
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    # styling (bold header, readable font)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.2)

    # bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_TABLE_PNG, dpi=250, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [compute_ttr_for_file(fp) for fp in sorted(INPUT_DIR.glob("*.txt"))]

    save_csv(rows)
    save_sorted_bar_png(rows)
    save_sorted_table_png(rows, max_rows=None)  # set to e.g. 40 if you prefer

    print(f"Done. CSV:\n  {OUTPUT_CSV}")
    print(f"Bar PNG:\n  {OUTPUT_BAR_PNG}")
    print(f"Table PNG:\n  {OUTPUT_TABLE_PNG}")


if __name__ == "__main__":
    main()
