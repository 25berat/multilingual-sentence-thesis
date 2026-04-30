# correlate_two_f1_tables_with_png.py

from __future__ import annotations

import os
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:
    raise SystemExit(
        "scipy fehlt. Installiere es z.B. mit: pip install scipy\n"
        f"Original error: {e}"
    )

try:
    import matplotlib.pyplot as plt
except Exception as e:
    raise SystemExit(
        "matplotlib fehlt. Installiere es z.B. mit: pip install matplotlib\n"
        f"Original error: {e}"
    )

EXCEL_PATH = r"C:\Users\berat\OneDrive\Dokumente\F1_matching.xlsx"
CSV_PATH   = r"C:\Users\berat\PycharmProjects\Belopsem\results\f1_table_deu_Latn_all_models.csv"

MODEL_COLS = ["glot500", "labse", "laser", "llama31", "qwen3", "sonar", "xlmr"]

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl == "lamma31":
            rename_map[c] = "llama31"
        elif cl in ("llama31", "glot500", "labse", "laser", "qwen3", "sonar", "xlmr"):
            rename_map[c] = cl
        elif cl in ("key", "src_lang", "trg_lang"):
            rename_map[c] = "key"
        elif cl in ("avg(key)", "avg_key", "avg"):
            rename_map[c] = "AVG(key)"
    return df.rename(columns=rename_map)

def _to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in MODEL_COLS + ["AVG(key)"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("-", pd.NA), errors="coerce")
    return df

def load_excel(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=0)
    df = _normalize_cols(df)

    if "key" not in df.columns:
        df = df.rename(columns={df.columns[0]: "key"})
    df["key"] = df["key"].astype(str).str.strip()

    keep = ["key"] + [c for c in MODEL_COLS if c in df.columns] + (["AVG(key)"] if "AVG(key)" in df.columns else [])
    df = _to_numeric(df[keep].copy())

    if "AVG(key)" not in df.columns:
        present_models = [c for c in MODEL_COLS if c in df.columns]
        df["AVG(key)"] = df[present_models].mean(axis=1, skipna=True) if present_models else pd.NA

    df = df[~df["key"].str.lower().isin(["avg(model)"])].copy()
    return df

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_cols(df)

    if "key" not in df.columns:
        raise ValueError("CSV muss eine Spalte 'key' (oder src_lang/trg_lang) haben.")

    df["key"] = df["key"].astype(str).str.strip()
    df = df[~df["key"].str.lower().eq("avg(model)")].copy()

    keep = ["key"] + [c for c in MODEL_COLS if c in df.columns] + (["AVG(key)"] if "AVG(key)" in df.columns else [])
    df = _to_numeric(df[keep].copy())

    if "AVG(key)" not in df.columns:
        present_models = [c for c in MODEL_COLS if c in df.columns]
        df["AVG(key)"] = df[present_models].mean(axis=1, skipna=True) if present_models else pd.NA

    return df

def corr_pairwise(x: pd.Series, y: pd.Series):
    d = pd.concat([x, y], axis=1).dropna()
    n = len(d)
    if n < 3:
        return d, {"n": n, "pearson_r": pd.NA, "pearson_p": pd.NA, "spearman_r": pd.NA, "spearman_p": pd.NA}

    xr = d.iloc[:, 0].astype(float).to_numpy()
    yr = d.iloc[:, 1].astype(float).to_numpy()

    pr, pp = pearsonr(xr, yr)
    sr, sp = spearmanr(xr, yr)
    return d, {"n": n, "pearson_r": pr, "pearson_p": pp, "spearman_r": sr, "spearman_p": sp}

def save_scatter_png(d_xy: pd.DataFrame, model: str, stats: dict, out_png: str):
    # d_xy has 2 columns: [excel, csv] (already dropna)
    x = d_xy.iloc[:, 0].astype(float).to_numpy()
    y = d_xy.iloc[:, 1].astype(float).to_numpy()

    plt.figure()
    plt.scatter(x, y)

    plt.xlabel(f"{model} (Excel)")
    plt.ylabel(f"{model} (CSV)")
    plt.title(f"Excel vs CSV — {model}")

    # y=x reference line (nice for seeing bias)
    if len(x) > 0 and len(y) > 0:
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        plt.plot([lo, hi], [lo, hi])

    # annotate
    n = stats.get("n", 0)
    pr = stats.get("pearson_r", pd.NA)
    sr = stats.get("spearman_r", pd.NA)
    text = f"n={n}\npearson r={pr:.4f}\nspearman ρ={sr:.4f}" if n >= 3 else f"n={n} (too small)"
    plt.gca().text(0.02, 0.98, text, transform=plt.gca().transAxes, va="top")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Excel nicht gefunden: {EXCEL_PATH}")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV nicht gefunden: {CSV_PATH}")

    df_x = load_excel(EXCEL_PATH)
    df_c = load_csv(CSV_PATH)

    merged = df_x.merge(df_c, on="key", how="inner", suffixes=("_excel", "_csv"))

    results = []

    # output folder for plots
    base_dir = os.path.dirname(CSV_PATH)
    plots_dir = os.path.join(base_dir, "plots_excel_vs_csv")
    os.makedirs(plots_dir, exist_ok=True)

    for m in MODEL_COLS:
        xcol = f"{m}_excel"
        ycol = f"{m}_csv"
        if xcol not in merged.columns or ycol not in merged.columns:
            continue

        d_xy, stats = corr_pairwise(merged[xcol], merged[ycol])
        results.append({"model": m, **stats})

        out_png = os.path.join(plots_dir, f"{m}.png")
        save_scatter_png(d_xy, m, stats, out_png)

    # AVG(key)
    if "AVG(key)_excel" in merged.columns and "AVG(key)_csv" in merged.columns:
        d_xy, stats = corr_pairwise(merged["AVG(key)_excel"], merged["AVG(key)_csv"])
        results.append({"model": "AVG(key)", **stats})

        out_png = os.path.join(plots_dir, "AVG_key.png")
        save_scatter_png(d_xy, "AVG(key)", stats, out_png)

    out = pd.DataFrame(results, columns=["model", "n", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])

    print("\n=== Correlation (Excel vs CSV), pairwise skip missing ===")
    print(out.to_string(index=False))

    out_csv = os.path.join(base_dir, "correlation_excel_vs_csv.csv")
    out.to_csv(out_csv, index=False)
    print(f"\n[Saved CSV]  {out_csv}")
    print(f"[Saved PNGs] {plots_dir}\\*.png")

if __name__ == "__main__":
    main()
