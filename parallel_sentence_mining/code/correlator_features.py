#!/usr/bin/env python3
"""
Compute Pearson correlations between ONE-HOT language features and F1 scores (per model).

NEW (aligned with regression script):
- No LANG, no VARIANT.
- Fixed inputs:
    Features (one-hot):  C:\\Users\\berat\\OneDrive\\Dokumente\\Lang_Features_encoded_onehot.xlsx
    F1 (all models):     C:\\Users\\berat\\PycharmProjects\\Belopsem\\results\\f1_table_deu_Latn_all_models.csv
- Handles mismatch in language counts by using only the overlap (inner join).
- Outputs:
    <Belopsem>/results/correlations/one_hot_encoding/<model_sanitized>/...
"""

from __future__ import annotations

import re
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------
# FIXED PATHS (same spirit as regression)
# --------------------------
FEATURE_XLSX = Path(r"C:\Users\berat\PycharmProjects\Belopsem\results\correlations\one_hot_encoding\Lang_Features_encoded_onehot.xlsx")
F1_ALL_MODELS_CSV = Path(r"C:\Users\berat\PycharmProjects\Belopsem\results\f1_table_deu_Latn_all_models.csv")

# Output root (inside Belopsem)
OUT_ROOT = F1_ALL_MODELS_CSV.parents[1] / "results" / "correlations" / "one_hot_encoding"

# correlation filtering
MIN_NON_NA = 6            # minimum paired samples to compute corr
MIN_FEATURE_STD = 1e-12   # feature must have some variance


# --------------------------
# HELPERS
# --------------------------
def sanitize_name(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s)
    return s[:120] if len(s) > 120 else s


def safe_strip_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM
        .str.strip()
    )


def find_id_column(df: pd.DataFrame) -> str:
    """
    Pick the best 'language id' column that exists in df.
    """
    candidates = ["lang", "language", "iso639_3", "iso", "id"]
    lower_map = {c.lower(): c for c in df.columns}
    for k in candidates:
        if k in lower_map:
            return lower_map[k]
    return df.columns[0]


def ensure_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def pearson_corr(x: pd.Series, y: pd.Series) -> float:
    """
    Pearson correlation with basic guards.
    """
    x = ensure_float_series(x)
    y = ensure_float_series(y)

    mask = x.notna() & y.notna()
    if int(mask.sum()) < MIN_NON_NA:
        return np.nan

    x2 = x[mask].values
    y2 = y[mask].values

    if float(np.std(x2)) < MIN_FEATURE_STD or float(np.std(y2)) < MIN_FEATURE_STD:
        return np.nan

    return float(np.corrcoef(x2, y2)[0, 1])


def is_onehot(col: str) -> bool:
    return "__" in col


def split_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    """
    One-hot columns look like: <group>__<category>
    We'll group by the part before '__'.
    """
    groups: Dict[str, List[str]] = {}
    for c in feature_cols:
        if "__" in c:
            grp = c.split("__", 1)[0]
            groups.setdefault(grp, []).append(c)
    return groups


def save_barplot(df_corr: pd.DataFrame, out_png: Path, title: str, max_bars: int = 60) -> None:
    """
    Save a barplot of correlation values (sorted). No seaborn.
    """
    if df_corr.empty:
        return

    tmp = df_corr.dropna(subset=["corr"]).copy()
    if tmp.empty:
        return

    tmp = tmp.sort_values("corr", ascending=False)

    if len(tmp) > max_bars:
        # keep strongest positives + strongest negatives
        top = tmp.head(max_bars // 2)
        bot = tmp.tail(max_bars // 2)
        tmp = pd.concat([top, bot], axis=0).sort_values("corr", ascending=False)

    labels = tmp["feature"].tolist()
    values = tmp["corr"].tolist()

    plt.figure(figsize=(12, max(6, 0.18 * len(labels))))
    plt.barh(range(len(labels)), values)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    plt.axvline(0.0, linewidth=1)
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_table_png(df_corr: pd.DataFrame, out_png: Path, title: str, n_top: int = 12, n_bottom: int = 12) -> None:
    """
    Save a simple table image of top/bottom correlations.
    """
    tmp = df_corr.dropna(subset=["corr"]).copy()
    if tmp.empty:
        return

    tmp = tmp.sort_values("corr", ascending=False)
    top = tmp.head(n_top)
    bottom = tmp.tail(n_bottom).sort_values("corr", ascending=True)

    show = pd.concat([top, bottom], axis=0).copy()
    show["corr"] = show["corr"].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")

    fig_h = max(4, 0.35 * (len(show) + 2))
    plt.figure(figsize=(12, fig_h))
    plt.axis("off")
    plt.title(title, fontsize=12)

    table = plt.table(
        cellText=show[["feature", "corr", "n"]].values,
        colLabels=["feature", "corr", "n"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# --------------------------
# MAIN
# --------------------------
def main():
    print("============================================================")
    print("ONE-HOT FEATURE ↔ F1 CORRELATIONS (ALL MODELS)")
    print("------------------------------------------------------------")
    print("[FEATURES] ", FEATURE_XLSX)
    print("[F1]       ", F1_ALL_MODELS_CSV)
    print("[OUT]      ", OUT_ROOT)
    print("============================================================\n")

    if not FEATURE_XLSX.is_file():
        raise FileNotFoundError(f"Feature file not found: {FEATURE_XLSX}")
    if not F1_ALL_MODELS_CSV.is_file():
        raise FileNotFoundError(f"F1 table not found: {F1_ALL_MODELS_CSV}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    feat_df = pd.read_excel(FEATURE_XLSX)
    f1_df = pd.read_csv(F1_ALL_MODELS_CSV, sep=None, engine="python")

    feat_df.columns = [str(c).strip() for c in feat_df.columns]
    f1_df.columns = [str(c).strip() for c in f1_df.columns]

    feat_id_col = find_id_column(feat_df)
    f1_id_col = "trg_lang" if "trg_lang" in f1_df.columns else find_id_column(f1_df)

    # normalize ids to string for merge
    feat_df[feat_id_col] = safe_strip_series(feat_df[feat_id_col])
    f1_df[f1_id_col] = safe_strip_series(f1_df[f1_id_col])

    # Overlap only (same as regression)
    excel_langs = set(feat_df[feat_id_col].tolist())
    f1_langs = set(f1_df[f1_id_col].tolist())
    common_langs = sorted(excel_langs & f1_langs)

    merged = feat_df.merge(
        f1_df,
        left_on=feat_id_col,
        right_on=f1_id_col,
        how="inner",
        suffixes=("", "_f1"),
    )

    print(f"[LANGS] Excel={len(excel_langs)} | F1={len(f1_langs)} | Overlap={len(common_langs)}")
    print(f"[MERGE] feature rows={len(feat_df)} | f1 rows={len(f1_df)} | merged rows={len(merged)}")

    if len(merged) < MIN_NON_NA:
        print("[WARN] Very few merged rows; correlations may be mostly NaN.")

    # Identify model columns: everything in f1_df except id, that is numeric-enough in merged
    candidate_model_cols = [c for c in f1_df.columns if c != f1_id_col]
    model_cols: List[str] = []
    for c in candidate_model_cols:
        test = pd.to_numeric(merged[c], errors="coerce")
        if int(test.notna().sum()) >= MIN_NON_NA:
            model_cols.append(c)

    if not model_cols:
        raise RuntimeError("No usable model columns found in F1 table (numeric with enough non-NA).")

    print(f"[MODELS] {model_cols}")

    # Feature columns: all columns from feat_df except id
    feature_cols = [c for c in feat_df.columns if c != feat_id_col]

    # Keep numeric-like features only (including 0/1 onehots + numeric features)
    numeric_feature_cols: List[str] = []
    for c in feature_cols:
        s = pd.to_numeric(merged[c], errors="coerce")
        if int(s.notna().sum()) >= MIN_NON_NA:
            numeric_feature_cols.append(c)

    onehot_groups = split_feature_groups([c for c in numeric_feature_cols if is_onehot(c)])
    print(f"[FEATURES] numeric-like features={len(numeric_feature_cols)} | onehot groups={len(onehot_groups)}\n")

    # Per model
    for model in model_cols:
        model_safe = sanitize_name(model)
        model_dir = OUT_ROOT / model_safe
        model_dir.mkdir(parents=True, exist_ok=True)

        y = ensure_float_series(merged[model])

        rows = []
        for feat in numeric_feature_cols:
            x = ensure_float_series(merged[feat])
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            corr = pearson_corr(x, y) if n >= MIN_NON_NA else np.nan
            rows.append({"feature": feat, "corr": corr, "n": n})

        corr_df = pd.DataFrame(rows).sort_values("corr", ascending=False, na_position="last")

        # main outputs
        out_csv = model_dir / f"feature_f1_correlations_{model_safe}.csv"
        out_png = model_dir / f"feature_f1_correlations_{model_safe}.png"
        out_tbl = model_dir / f"feature_f1_correlations_{model_safe}_table.png"

        corr_df.to_csv(out_csv, index=False, encoding="utf-8")
        save_barplot(corr_df, out_png, title=f"one-hot features vs {model} (overlap n={len(merged)})")
        save_table_png(corr_df, out_tbl, title=f"top/bottom correlations vs {model} (overlap n={len(merged)})")

        # ---- all_features subsets ----
        all_dir = model_dir / "all_features"
        all_dir.mkdir(parents=True, exist_ok=True)

        corr_df.to_csv(all_dir / "all_features.csv", index=False, encoding="utf-8")
        save_barplot(corr_df, all_dir / "all_features.png", title=f"all features vs {model} (n={len(merged)})")

        def _range_df(lo: float, hi: float) -> pd.DataFrame:
            tmp = corr_df.dropna(subset=["corr"]).copy()
            return tmp[(tmp["corr"] >= lo) & (tmp["corr"] <= hi)].sort_values("corr", ascending=False)

        r1 = _range_df(-1.0, -0.3)
        r2 = _range_df(-0.3, 0.3)
        r3 = _range_df(0.3, 1.0)

        r1.to_csv(all_dir / "range_neg1_neg0_3.csv", index=False, encoding="utf-8")
        r2.to_csv(all_dir / "range_neg0_3_pos0_3.csv", index=False, encoding="utf-8")
        r3.to_csv(all_dir / "range_pos0_3_pos1.csv", index=False, encoding="utf-8")

        if not r1.empty:
            save_barplot(r1, all_dir / "range_neg1_neg0_3.png", title=f"corr in [-1,-0.3] vs {model} (n={len(merged)})")
        if not r2.empty:
            save_barplot(r2, all_dir / "range_neg0_3_pos0_3.png", title=f"corr in [-0.3,0.3] vs {model} (n={len(merged)})")
        if not r3.empty:
            save_barplot(r3, all_dir / "range_pos0_3_pos1.png", title=f"corr in [0.3,1] vs {model} (n={len(merged)})")

        # ---- numeric only (no "__") ----
        numeric_only = corr_df[~corr_df["feature"].map(is_onehot)].copy()
        num_dir = model_dir / "numeric"
        num_dir.mkdir(parents=True, exist_ok=True)
        numeric_only.to_csv(num_dir / "numeric_all.csv", index=False, encoding="utf-8")
        if not numeric_only.dropna(subset=["corr"]).empty:
            save_barplot(numeric_only, num_dir / "numeric_all.png", title=f"numeric-only vs {model} (n={len(merged)})")

        # ---- onehot groups ----
        groups_root = model_dir / "onehot_groups"
        groups_root.mkdir(parents=True, exist_ok=True)

        for grp, cols in onehot_groups.items():
            grp_safe = sanitize_name(grp)
            grp_dir = groups_root / grp_safe
            grp_dir.mkdir(parents=True, exist_ok=True)

            gdf = corr_df[corr_df["feature"].isin(cols)].copy()
            gdf.to_csv(grp_dir / f"group_{grp_safe}.csv", index=False, encoding="utf-8")
            if not gdf.dropna(subset=["corr"]).empty:
                save_barplot(gdf, grp_dir / f"group_{grp_safe}.png", title=f"group={grp} vs {model} (n={len(merged)})", max_bars=80)

        print(f"[DONE] Model '{model}' -> {model_dir}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
