#!/usr/bin/env python3
import csv
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# CHANGE ONLY THIS:
LANG = "deu_Latn"   # or: "tur_Latn"

# results tree:
#   <PROJECT_ROOT>/results/<LANG>/<variant>/<model>/...
RESULTS_ROOT = PROJECT_ROOT / "results" / LANG

# variants to process (skips if missing)
VARIANTS = ["raw"]  # or ["raw"] or ["whitened"]

# =========================
# CHANGE THIS WHEN NEEDED
# If None -> process ALL models under variant/
# Example: ["llama31_noprompt"] or ["llama31_noprompt", "llama31_noprompt"]
# =========================
#HARD_MODELS = ["glot500","labse", "laser", "llama31", "qwen3", "sonar", "xlmr"]
HARD_MODELS = ["studentteacher", "glot500","labse"]

# =========================

# Match numeric line "P,R,F1"
NUM_LINE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*$")

# Optional aliases (if your folders sometimes use short ISO codes)
LANG_ALIASES = {
    "deu_Latn": ["deu_Latn", "de"],
    "tur_Latn": ["tur_Latn", "tur"],
}
LANG_SRC_ACCEPT = set(LANG_ALIASES.get(LANG, [LANG]))


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


def collect_model_f1(model_dir: Path) -> Tuple[str, Dict[str, float], str]:
    """
    Returns:
      model_name, {trg_lang: f1}, pivot_col_name (LANG or alias actually seen)

    We look in:
      <model_dir>/mining/bucc2017/<src>-<trg>/*.res
    and only take pairs where src == LANG (or alias).
    """
    model_name = model_dir.name  # IMPORTANT: use folder name
    mining_root = model_dir / "mining" / "bucc2017"
    trg_to_f1: Dict[str, float] = {}
    pivot_col = LANG  # column label (we may overwrite if alias "de"/"tur" is seen)

    if not mining_root.is_dir():
        return model_name, trg_to_f1, pivot_col

    for pair_dir in sorted(p for p in mining_root.iterdir() if p.is_dir()):
        pair = pair_dir.name
        if "-" not in pair:
            continue
        src, trg = pair.split("-", 1)

        # only pivot-as-source (LANG)
        if src not in LANG_SRC_ACCEPT:
            continue

        pivot_col = src  # keep what was actually found in folder naming

        # find a .res
        res_files = sorted(pair_dir.glob("*.pred.res"))
        if not res_files:
            res_files = sorted(pair_dir.glob("*.res"))

        f1_val = None
        for rf in res_files:
            f1 = read_f1_from_res(rf)
            if f1 is not None:
                f1_val = f1
                break

        if f1_val is not None:
            trg_to_f1.setdefault(trg, f1_val)

    return model_name, trg_to_f1, pivot_col


def write_model_csv(model_dir: Path, rows: List[Tuple[str, float]], pivot_col: str) -> Path:
    out_csv = model_dir / f"f1_table_{LANG}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trg_lang", pivot_col])
        for trg, f1 in rows:
            w.writerow([trg, f1])
    return out_csv


def render_table_png(cell_rows: List[List[str]],
                     row_labels: List[str],
                     col_labels: List[str],
                     title: str,
                     out_png: Path) -> None:
    n_rows = len(row_labels)
    n_cols = len(col_labels)
    fig_w = max(6, min(22, 2 + 0.9 * n_cols))
    fig_h = max(4, min(22, 1.8 + 0.35 * n_rows))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=cell_rows,
        rowLabels=row_labels,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.1, 1.1)
    ax.set_title(title, pad=12, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_per_model_outputs(model_dir: Path, trg_to_f1: Dict[str, float], pivot_col: str, variant: str) -> None:
    if not trg_to_f1:
        print(f"[skip] {LANG}/{variant}/{model_dir.name}: no F1 entries.")
        return

    rows = sorted(trg_to_f1.items(), key=lambda x: x[0])

    out_csv = write_model_csv(model_dir, rows, pivot_col)

    cell_rows = [[f"{f1:.4f}"] for _, f1 in rows]
    row_labels = [trg for trg, _ in rows]
    col_labels = [pivot_col]
    out_png = model_dir / f"f1_table_{LANG}.png"

    render_table_png(
        cell_rows,
        row_labels,
        col_labels,
        title=f"F1 vs {pivot_col} ({LANG}/{variant}/{model_dir.name})",
        out_png=out_png
    )

    print(f"[done] {LANG}/{variant}/{model_dir.name}: wrote {out_csv.name} and {out_png.name}")


def make_combined_outputs(all_models: Dict[str, Dict[str, float]], variant: str) -> None:
    """
    One big table per variant:
      rows = target langs
      columns = models under this variant
    """
    if not all_models:
        print(f"[info] {LANG}/{variant}: nothing to combine.")
        return

    all_trg = sorted({t for d in all_models.values() for t in d.keys()})
    all_model_names = sorted(all_models.keys())

    matrix: List[List[str]] = []
    row_avgs: List[float] = []

    for trg in all_trg:
        row_vals: List[float] = []
        for m in all_model_names:
            v = all_models[m].get(trg)
            row_vals.append(v if v is not None else float("nan"))

        present = [v for v in row_vals if not (v != v)]
        row_avg = sum(present) / len(present) if present else float("nan")
        row_avgs.append(row_avg)

        row_str = []
        for v in row_vals:
            row_str.append("" if (v != v) else f"{v:.4f}")
        row_str.append("" if (row_avg != row_avg) else f"{row_avg:.4f}")
        matrix.append(row_str)

    # column averages (avg per model)
    col_avgs: List[str] = []
    for m in all_model_names:
        vals = [v for v in all_models[m].values() if v is not None]
        col_avgs.append(f"{(sum(vals)/len(vals)):.4f}" if vals else "")

    bottom_cells = col_avgs + [""]

    out_csv = RESULTS_ROOT / variant / f"f1_table_{LANG}_all_models.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trg_lang"] + all_model_names + ["AVG(trg)"])
        for i, trg in enumerate(all_trg):
            w.writerow([trg] + matrix[i])
        w.writerow(["AVG(model)"] + col_avgs + [""])

    row_labels = all_trg + ["AVG(model)"]
    col_labels = all_model_names + ["AVG(trg)"]
    matrix.append(bottom_cells)

    out_png = RESULTS_ROOT / variant / f"f1_table_{LANG}_all_models.png"
    render_table_png(
        matrix,
        row_labels,
        col_labels,
        title=f"F1 vs {LANG} across models ({variant})",
        out_png=out_png
    )

    print(f"[done] {LANG}/{variant}: wrote {out_csv.name} and {out_png.name}")


def main():
    if not RESULTS_ROOT.is_dir():
        raise SystemExit(f"[error] results folder not found: {RESULTS_ROOT}")

    for variant in VARIANTS:
        variant_root = RESULTS_ROOT / variant
        if not variant_root.is_dir():
            print(f"[info] variant '{variant}' not found under {RESULTS_ROOT}, skipping.")
            continue

        print(f"[info] collecting F1 for: LANG={LANG}, variant={variant}")
        print(f"[info] hard_models = {HARD_MODELS}")

        all_models: Dict[str, Dict[str, float]] = {}

        # choose which model dirs to process
        if HARD_MODELS is None:
            model_dirs = sorted(p for p in variant_root.iterdir() if p.is_dir())
        else:
            model_dirs = [variant_root / name for name in HARD_MODELS]

        for model_dir in model_dirs:
            if not model_dir.exists():
                print(f"[warn] {LANG}/{variant}: model folder not found: {model_dir} (skipping)")
                continue
            if not model_dir.is_dir():
                continue

            model_name, trg_to_f1, pivot_col = collect_model_f1(model_dir)
            if trg_to_f1:
                make_per_model_outputs(model_dir, trg_to_f1, pivot_col, variant)
                all_models[model_name] = trg_to_f1
            else:
                print(f"[warn] {LANG}/{variant}/{model_name}: no pivot-source F1 found (src in {sorted(LANG_SRC_ACCEPT)}).")

        make_combined_outputs(all_models, variant)


if __name__ == "__main__":
    main()
