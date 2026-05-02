#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
BASE_DIR = Path(r"C:\Users\berat\PycharmProjects\outliers\src\centroids\raw")

DEU_TSV = BASE_DIR / "dist_from_deu_Latn_avg.tsv"
TUR_TSV = BASE_DIR / "dist_from_tur_Latn_avg.tsv"

OUT_PNG = BASE_DIR / "compare_dist_deu_Latn_vs_tur_Latn.png"

# Which columns to compare (must exist in TSV header)
# (Your TSV header is typically: lang, avg_distance, glot500, labse, sonar, laser, sonar)
COMPARE_COLS = ["avg_distance", "glot500", "labse", "sonar", "laser", "sonar"]

# Formatting
DECIMALS = 6


def read_tsv(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Optional[float]]]]:
    """
    Returns:
      cols: header columns (excluding 'lang')
      data: {lang: {col: float_or_None}}
    """
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if not header or header[0] != "lang":
            raise ValueError(f"Unexpected header in {path}: {header[:10]}")

        cols = header[1:]
        data: Dict[str, Dict[str, Optional[float]]] = {}

        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or len(parts) < 1:
                continue
            lang = parts[0]
            vals = parts[1:]

            row: Dict[str, Optional[float]] = {}
            for c, v in zip(cols, vals):
                v = v.strip()
                if v == "":
                    row[c] = None
                else:
                    try:
                        row[c] = float(v)
                    except ValueError:
                        row[c] = None
            data[lang] = row

    return cols, data


def fmt(v: Optional[float]) -> str:
    if v is None:
        return ""
    return f"{v:.{DECIMALS}f}"


def main():
    cols_deu, deu = read_tsv(DEU_TSV)
    cols_tur, tur = read_tsv(TUR_TSV)

    # ensure compare cols exist
    for c in COMPARE_COLS:
        if c not in cols_deu:
            raise SystemExit(f"[error] column '{c}' not found in {DEU_TSV.name}")
        if c not in cols_tur:
            raise SystemExit(f"[error] column '{c}' not found in {TUR_TSV.name}")

    # union of languages, alphabetical
    all_langs = sorted(set(deu.keys()) | set(tur.keys()))

    # Build table columns: for each metric/model -> deu + tur
    col_labels: List[str] = ["lang"]
    col_pairs: List[Tuple[str, str]] = []  # (col_name, ref) where ref in {"deu","tur"}
    for c in COMPARE_COLS:
        col_labels.append(f"{c} (deu)")
        col_labels.append(f"{c} (tur)")
        col_pairs.append((c, "deu"))
        col_pairs.append((c, "tur"))

    # Prepare cell text and colors
    cell_text: List[List[str]] = []
    cell_colors: List[List[Tuple[float, float, float, float]]] = []

    # Collect averages per column (excluding 'lang')
    # We'll store values separately for deu/tur per COMPARE_COLS
    values_deu: Dict[str, List[float]] = {c: [] for c in COMPARE_COLS}
    values_tur: Dict[str, List[float]] = {c: [] for c in COMPARE_COLS}

    # Color palette (RGBA)
    GREEN = (0.80, 0.93, 0.80, 1.0)   # light green
    RED   = (0.96, 0.80, 0.80, 1.0)   # light red
    GREY  = (0.92, 0.92, 0.92, 1.0)   # neutral
    WHITE = (1.00, 1.00, 1.00, 1.0)

    for lang in all_langs:
        row_txt = [lang]
        row_col = [WHITE]

        deu_row = deu.get(lang, {})
        tur_row = tur.get(lang, {})

        for c in COMPARE_COLS:
            vd = deu_row.get(c)
            vt = tur_row.get(c)

            # accumulate for avg row
            if isinstance(vd, float):
                values_deu[c].append(vd)
            if isinstance(vt, float):
                values_tur[c].append(vt)

            # decide coloring (per language+col compare)
            if vd is None and vt is None:
                cd, ct = WHITE, WHITE
            elif vd is None and vt is not None:
                cd, ct = WHITE, WHITE
            elif vt is None and vd is not None:
                cd, ct = WHITE, WHITE
            else:
                # both present
                if abs(vd - vt) < 1e-12:
                    cd, ct = GREY, GREY
                elif vd < vt:
                    # smaller distance = better (green)
                    cd, ct = GREEN, RED
                else:
                    cd, ct = RED, GREEN

            row_txt.append(fmt(vd))
            row_txt.append(fmt(vt))
            row_col.append(cd)
            row_col.append(ct)

        cell_text.append(row_txt)
        cell_colors.append(row_col)

    # Add AVG row
    avg_row_txt = ["AVG"]
    avg_row_col = [GREY]

    for c in COMPARE_COLS:
        avg_d = float(np.mean(values_deu[c])) if values_deu[c] else None
        avg_t = float(np.mean(values_tur[c])) if values_tur[c] else None

        # color avg comparison
        if avg_d is None or avg_t is None:
            cd, ct = GREY, GREY
        else:
            if abs(avg_d - avg_t) < 1e-12:
                cd, ct = GREY, GREY
            elif avg_d < avg_t:
                cd, ct = GREEN, RED
            else:
                cd, ct = RED, GREEN

        avg_row_txt.append(fmt(avg_d))
        avg_row_txt.append(fmt(avg_t))
        avg_row_col.append(cd)
        avg_row_col.append(ct)

    cell_text.append(avg_row_txt)
    cell_colors.append(avg_row_col)

    # Render matplotlib table
    n_rows = len(cell_text)
    n_cols = len(col_labels)

    # Figure size heuristics
    fig_w = max(12, min(28, 2 + 0.9 * n_cols))
    fig_h = max(6, min(40, 2 + 0.22 * n_rows))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=250)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.05)

    ax.set_title(
        "Distance comparison: dist_from_deu_Latn vs dist_from_tur_Latn (lower = better)",
        pad=14,
        fontsize=12,
    )

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
