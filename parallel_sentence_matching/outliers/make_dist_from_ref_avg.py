#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# CONFIG (change if needed)
# =========================
MODELS = ["glot500", "labse", "xlmr", "laser", "sonar", "llama31", "qwen3"]
VARIANT = "raw"
REF_LANG = "tur_Latn"

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parent                 # .../outliers/src
BASE_DIR = SRC_DIR / "centroids"      # .../outliers/src/centroids
VARIANT_DIR = BASE_DIR / VARIANT      # .../outliers/src/centroids/raw


def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def read_ref_row_from_distance_csv(csv_path: Path, ref_lang: str) -> Dict[str, float]:
    """
    Reads your distance matrix CSV:
      header: lang,<L1>,<L2>,...
      rows:   <lang_i>,d(i,1),d(i,2),...
    Returns dict {lang_j: dist(ref_lang, lang_j)} extracted from the row ref_lang.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        if len(header) < 2 or header[0] != "lang":
            raise ValueError(f"Unexpected header in {csv_path}: {header[:5]} ...")
        langs = header[1:]

        for line in f:
            parts = line.rstrip("\n").split(",")
            if not parts:
                continue
            lang = parts[0]
            if lang != ref_lang:
                continue
            vals = [float(x) for x in parts[1:]]
            if len(vals) != len(langs):
                raise ValueError(f"Row length mismatch in {csv_path} for {ref_lang}")
            return dict(zip(langs, vals))

    # not found
    return {}


def write_avg_tsv_and_table_png(
    ref_lang: str,
    per_model: Dict[str, Dict[str, float]],
    out_tsv: Path,
    out_table_png: Path,
) -> None:
    """
    per_model:
      {"sonar": {"deu_Latn": 7.1, ...}, "labse": {...}, ...}
    Writes:
      dist_from_<ref>_avg.tsv with columns: lang, avg_distance, <models...>
      dist_from_<ref>_avg_table.png as a table image
    """
    # union of langs across models
    all_langs = set()
    for d in per_model.values():
        all_langs.update(d.keys())
    all_langs = sorted(all_langs)

    model_cols = list(per_model.keys())  # keep the found order

    rows = []
    for lang in all_langs:
        vals = []
        row_model_vals = []
        for m in model_cols:
            v = per_model[m].get(lang)
            if v is None:
                row_model_vals.append("")
            else:
                vals.append(v)
                row_model_vals.append(v)

        # define avg
        if lang == ref_lang:
            avg = 0.0
        else:
            avg = float(sum(vals) / len(vals)) if vals else None

        rows.append((lang, avg, row_model_vals))

    # write TSV
    ensure_dir(out_tsv)
    with out_tsv.open("w", encoding="utf-8") as f:
        f.write("\t".join(["lang", "avg_distance"] + model_cols) + "\n")
        for lang, avg, mv in rows:
            line = [lang, f"{avg:.6f}" if isinstance(avg, float) else ""]
            for v in mv:
                if isinstance(v, float):
                    line.append(f"{v:.6f}")
                else:
                    line.append("")
            f.write("\t".join(line) + "\n")

    print(f"[✓] wrote → {out_tsv}")

    # table png
    # sort by avg distance (ascending), keep ref on top
    def sort_key(item):
        lang, avg, _ = item
        if lang == ref_lang:
            return (-1.0)  # very first
        if isinstance(avg, float):
            return avg
        return 1e18

    rows_sorted = sorted(rows, key=sort_key)

    cell_text = []
    for lang, avg, mv in rows_sorted:
        r = [lang, f"{avg:.6f}" if isinstance(avg, float) else ""]
        for v in mv:
            r.append(f"{v:.6f}" if isinstance(v, float) else "")
        cell_text.append(r)

    col_labels = ["lang", "avg_distance"] + model_cols

    # size heuristic
    fig_w = max(9, min(22, 2 + 1.2 * len(col_labels)))
    fig_h = max(4, min(28, 1.5 + 0.26 * len(rows_sorted)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=250)
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.05)
    ax.set_title(f"Distances from {ref_lang} (avg across models) - {VARIANT}", pad=12, fontsize=12)

    ensure_dir(out_table_png)
    plt.tight_layout()
    plt.savefig(out_table_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[✓] wrote → {out_table_png}")


def main():
    if not VARIANT_DIR.is_dir():
        raise SystemExit(f"[error] variant dir not found: {VARIANT_DIR}")

    per_model: Dict[str, Dict[str, float]] = {}

    for m in MODELS:
        csv_path = VARIANT_DIR / m / f"lang_distances_centroids_{m}.csv"
        if not csv_path.is_file():
            print(f"[warn] missing: {csv_path}")
            continue

        d = read_ref_row_from_distance_csv(csv_path, REF_LANG)
        if not d:
            print(f"[warn] {REF_LANG} not found in: {csv_path}")
            continue

        per_model[m] = d
        print(f"[ok] loaded ref-row from {m}: {csv_path}")

    if not per_model:
        raise SystemExit("[error] no models loaded (no ref-row found).")

    out_tsv = VARIANT_DIR / f"dist_from_{REF_LANG}_avg.tsv"
    out_table_png = VARIANT_DIR / f"dist_from_{REF_LANG}_avg_table.png"

    write_avg_tsv_and_table_png(
        ref_lang=REF_LANG,
        per_model=per_model,
        out_tsv=out_tsv,
        out_table_png=out_table_png,
    )


if __name__ == "__main__":
    main()
