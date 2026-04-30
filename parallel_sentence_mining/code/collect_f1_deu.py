#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

NUM_LINE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*$")

# Your possible mining roots (we try all)
CANDIDATE_PAIR_ROOTS = [
    ("MINING", "bucc"),
    ("mining", "bucc"),
    ("mining", "bucc2017"),
    ("MINING", "bucc2017"),
    ("MINING", "bucc2017"),  # harmless duplicate; keep if you add more
]


def read_f1_from_res(res_path: Path) -> Optional[float]:
    """Return the last F1 value in a .res file, or None."""
    if not res_path.is_file():
        return None
    last = None
    for line in res_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = NUM_LINE_RE.match(line.strip())
        if m:
            last = float(m.group(3))
    return last


def find_pairs_root(model_dir: Path) -> Optional[Path]:
    """
    Find the first existing root that contains pair dirs like <src>-<trg>.
    Example expected:
      results/<model>/MINING/bucc/deu_Latn-ace_Arab/...
    """
    for a, b in CANDIDATE_PAIR_ROOTS:
        p = model_dir / a / b
        if p.is_dir():
            return p
    return None


def list_pair_dirs(pairs_root: Path) -> List[Path]:
    return sorted([p for p in pairs_root.iterdir() if p.is_dir() and "-" in p.name])


def detect_pivot(results_root: Path) -> Optional[str]:
    """
    Detect pivot as most common src from pair dir names '<src>-<trg>' across all models.
    """
    ctr = Counter()
    for model_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        pairs_root = find_pairs_root(model_dir)
        if not pairs_root:
            continue
        for pair_dir in list_pair_dirs(pairs_root):
            src, _trg = pair_dir.name.split("-", 1)
            ctr[src] += 1
    return ctr.most_common(1)[0][0] if ctr else None


def collect_model_f1(model_dir: Path, pivot: Optional[str]) -> Tuple[str, Dict[str, float]]:
    """
    Collect F1 per target language for this model.
    If pivot is provided: only pairs where src == pivot.
    If pivot is None: we keep the whole pair_dir.name as key.
    """
    model_name = model_dir.name
    pairs_root = find_pairs_root(model_dir)
    out: Dict[str, float] = {}

    if not pairs_root:
        return model_name, out

    for pair_dir in list_pair_dirs(pairs_root):
        pair_name = pair_dir.name
        src, trg = pair_name.split("-", 1)

        if pivot is not None and src != pivot:
            continue

        # prefer *.pred.res, else *.res
        res_files = sorted(pair_dir.glob("*.pred.res"))
        if not res_files:
            res_files = sorted(pair_dir.glob("*.res"))

        f1_val = None
        for rf in res_files:
            f1 = read_f1_from_res(rf)
            if f1 is not None:
                f1_val = f1
                break

        if f1_val is None:
            continue

        key = trg if pivot is not None else pair_name
        out[key] = f1_val

    return model_name, out


def render_table_png(cell_rows, row_labels, col_labels, title, out_png: Path) -> None:
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    fig_w = max(6, min(26, 2 + 0.9 * n_cols))
    fig_h = max(4, min(26, 1.8 + 0.35 * n_rows))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=cell_rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.1)
    ax.set_title(title, pad=12, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_per_model_outputs(model_dir: Path, pivot_label: str, key_to_f1: Dict[str, float]) -> None:
    if not key_to_f1:
        print(f"[skip] {model_dir.name}: no F1 entries.")
        return

    rows = sorted(key_to_f1.items(), key=lambda x: x[0])

    out_csv = model_dir / f"f1_table_{pivot_label}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", pivot_label])
        for k, f1 in rows:
            w.writerow([k, f"{f1:.6f}"])

    out_png = model_dir / f"f1_table_{pivot_label}.png"
    render_table_png(
        cell_rows=[[f"{f1:.4f}"] for _, f1 in rows],
        row_labels=[k for k, _ in rows],
        col_labels=[pivot_label],
        title=f"F1 table ({model_dir.name})",
        out_png=out_png,
    )

    print(f"[done] {model_dir.name}: wrote {out_csv.name} and {out_png.name}")


def write_combined_outputs(pivot_label: str, all_models: Dict[str, Dict[str, float]]) -> None:
    if not all_models:
        print("[info] nothing to combine.")
        return

    keys = sorted({k for d in all_models.values() for k in d.keys()})
    models = sorted(all_models.keys())

    matrix: List[List[str]] = []
    for k in keys:
        vals = []
        for m in models:
            v = all_models[m].get(k)
            vals.append(v if v is not None else float("nan"))
        present = [v for v in vals if not (v != v)]
        avg = sum(present) / len(present) if present else float("nan")

        row = [("" if (v != v) else f"{v:.4f}") for v in vals]
        row.append("" if (avg != avg) else f"{avg:.4f}")
        matrix.append(row)

    # model averages
    model_avgs = []
    for m in models:
        mv = list(all_models[m].values())
        model_avgs.append(f"{(sum(mv)/len(mv)):.4f}" if mv else "")
    bottom = model_avgs + [""]

    out_csv = RESULTS_ROOT / f"f1_table_{pivot_label}_all_models.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key"] + models + ["AVG(key)"])
        for k, row in zip(keys, matrix):
            w.writerow([k] + row)
        w.writerow(["AVG(model)"] + bottom)

    out_png = RESULTS_ROOT / f"f1_table_{pivot_label}_all_models.png"
    render_table_png(
        cell_rows=matrix + [bottom],
        row_labels=keys + ["AVG(model)"],
        col_labels=models + ["AVG(key)"],
        title=f"F1 across models ({pivot_label})",
        out_png=out_png,
    )

    print(f"[done] combined: wrote {out_csv.name} and {out_png.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pivot", default=None, help="Optional pivot src tag, e.g. deu_Latn. If omitted, auto-detect.")
    ap.add_argument("--no-pivot", action="store_true", help="Ignore src/trg and just use full pair name as key.")
    args = ap.parse_args()

    if not RESULTS_ROOT.is_dir():
        raise SystemExit(f"[error] results folder not found: {RESULTS_ROOT}")

    if args.no_pivot:
        pivot = None
        pivot_label = "all_pairs"
    else:
        pivot = args.pivot or detect_pivot(RESULTS_ROOT)
        if not pivot:
            raise SystemExit(
                "[error] Could not auto-detect pivot. "
                "Use --no-pivot or pass --pivot <SRC_TAG>."
            )
        pivot_label = pivot

    print(f"[info] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[info] RESULTS_ROOT:  {RESULTS_ROOT}")
    print(f"[info] mode:         {'no-pivot' if pivot is None else 'pivot=' + pivot}")

    all_models: Dict[str, Dict[str, float]] = {}
    for model_dir in sorted([p for p in RESULTS_ROOT.iterdir() if p.is_dir()]):
        pairs_root = find_pairs_root(model_dir)
        if not pairs_root:
            print(f"[skip] {model_dir.name}: missing MINING/bucc (or other candidates)")
            continue

        model_name, key_to_f1 = collect_model_f1(model_dir, pivot=pivot)
        write_per_model_outputs(model_dir, pivot_label=pivot_label, key_to_f1=key_to_f1)
        if key_to_f1:
            all_models[model_name] = key_to_f1

    write_combined_outputs(pivot_label=pivot_label, all_models=all_models)


if __name__ == "__main__":
    main()
