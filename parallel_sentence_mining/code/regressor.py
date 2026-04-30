#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless (no GUI)
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import KFold, cross_val_score


# ============================================================
# CONFIG (CHANGE HERE)
# ============================================================

# Choose your model here:
#   "labse" | "glot500" | "sonar" | "qwen3" | "llama31" | "xlmr" | "laser"
EMB_MODEL = "llama31"

# Your features file (raw, not onehot-encoded)
FEATURES_XLSX = Path(r"C:\Users\berat\OneDrive\Dokumente\Lang_Features.xlsx")

# Fixed F1 input (all models in one CSV)
F1_ALL_MODELS_CSV = Path(r"C:\Users\berat\PycharmProjects\Belopsem\results\f1_table_deu_Latn_all_models.csv")

# ElasticNet sparsity target (approx)
TARGET_SELECTED_FEATURES = 10

# CV settings
CV_FOLDS = 5
RANDOM_STATE = 0

# Random test set size
N_TEST = 6

# Clip predictions to [0,1] (because F1)
CLIP_TO_UNIT = True

# Make a PNG for easy reading of predicted vs true
MAKE_PNG = True


# ============================================================
# MODEL SCHEMA (EDIT HERE IF YOUR COLUMN NAMES DIFFER)
# ============================================================
# This tells the script which:
# - column in the F1 CSV contains the target F1 for this model (f1_col)
# - distance feature column in Excel to keep (dist_col)
# - training-flag column in Excel to keep (train_flag_col)
# - optional model-specific TTR column in Excel to keep (ttr_col)
#
# If some of these columns do not exist in your Excel, that's OK:
# we will just not use them, and we will not crash.
MODEL_SPECS: Dict[str, Dict[str, Optional[str]]] = {
    "labse":   {"f1_col": "labse",   "dist_col": "dist_deu_labse",   "train_flag_col": "is_training_lang_labse",   "ttr_col": "TTR_labse"},
    "glot500": {"f1_col": "glot500", "dist_col": "dist_deu_glot500", "train_flag_col": "is_training_lang_glot500", "ttr_col": "TTR_glot500"},
    "sonar":   {"f1_col": "sonar",   "dist_col": "dist_deu_sonar",   "train_flag_col": "is_training_lang_sonar",   "ttr_col": None},

    # New ones you asked for (adjust names here if your CSV/Excel uses different spellings)
    "qwen3":   {"f1_col": "qwen3",   "dist_col": "dist_deu_qwen3",   "train_flag_col": "is_training_lang_qwen3",   "ttr_col": "TTR_qwen3"},
    "llama31": {"f1_col": "llama31", "dist_col": "dist_deu_llama31", "train_flag_col": "is_training_lang_llama31", "ttr_col": "TTR_llama31"},
    "xlmr":    {"f1_col": "xlmr",    "dist_col": "dist_deu_xlmr",    "train_flag_col": "is_training_lang_xlmr",    "ttr_col": "TTR_xlmr"},
    "laser":   {"f1_col": "laser",   "dist_col": "dist_deu_laser",   "train_flag_col": "is_training_lang_laser",   "ttr_col": "TTR_laser"},
}


# ============================================================
# HELPERS
# ============================================================

def parse_numeric_or_percent(x: Any) -> float:
    """
    Robust parsing for numbers that might contain commas, spaces, percent signs, etc.
    NOTE: we do NOT convert % to 0-1 here.
    """
    if x is None:
        return math.nan

    if isinstance(x, float):
        return x

    if isinstance(x, int):
        return float(x)

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return math.nan

        if s.endswith("%"):
            s = s[:-1].strip()

        # German decimals "1,23"
        if "," in s and "." not in s:
            s = s.replace(",", ".")

        # remove spaces
        s = s.replace("\u202f", "").replace(" ", "")

        try:
            return float(s)
        except ValueError:
            return math.nan

    return math.nan


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_model_spec(model_name: str) -> Dict[str, Optional[str]]:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"Unknown EMB_MODEL='{model_name}'. Allowed: {sorted(MODEL_SPECS.keys())}")
    return MODEL_SPECS[model_name]


def build_outputs(base_dir: Path, emb_model: str) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """
    Output directory + files.
    """
    out_dir = base_dir / "results" / "regression" / emb_model
    ensure_dir(out_dir)

    pred_csv = out_dir / f"pred_random{N_TEST}_{emb_model}_elasticnet.csv"
    coef_all_csv = out_dir / f"coefficients_ALL_{emb_model}.csv"
    coef_selected_csv = out_dir / f"coefficients_SELECTED_{emb_model}.csv"
    report_txt = out_dir / f"model_report_{emb_model}.txt"
    pred_vs_true_csv = out_dir / f"pred_vs_true_random{N_TEST}_{emb_model}.csv"
    compare_png = out_dir / f"compare_pred_true_random{N_TEST}_{emb_model}.png"

    return out_dir, pred_csv, coef_all_csv, coef_selected_csv, report_txt, pred_vs_true_csv, compare_png


def safe_strip_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace("\ufeff", "", regex=False)  # remove BOM
        .str.strip()
    )


def main():
    spec = get_model_spec(EMB_MODEL)
    f1_col = spec["f1_col"]
    dist_col_keep = spec["dist_col"]
    train_flag_keep = spec["train_flag_col"]
    ttr_model_col = spec["ttr_col"]

    # Base dir = Belopsem project root inferred from the fixed CSV path
    base_dir = F1_ALL_MODELS_CSV.parents[1]  # ...\Belopsem
    out_dir, pred_csv, coef_all_csv, coef_selected_csv, report_txt, pred_vs_true_csv, compare_png = build_outputs(base_dir, EMB_MODEL)

    print("============================================================")
    print("REGRESSION CONFIG")
    print("------------------------------------------------------------")
    print("EMB_MODEL         :", EMB_MODEL)
    print("F1 model column   :", f1_col)
    print("FEATURES_XLSX     :", FEATURES_XLSX)
    print("F1_ALL_MODELS_CSV :", F1_ALL_MODELS_CSV)
    print("OUT_DIR           :", out_dir)
    print("N_TEST            :", N_TEST)
    print("RANDOM_STATE      :", RANDOM_STATE)
    print("------------------------------------------------------------")
    print("Keep dist col     :", dist_col_keep)
    print("Keep train flag   :", train_flag_keep)
    print("Keep TTR model col:", ttr_model_col)
    print("============================================================\n")

    if not FEATURES_XLSX.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_XLSX}")
    if not F1_ALL_MODELS_CSV.exists():
        raise FileNotFoundError(f"F1 all-models file not found: {F1_ALL_MODELS_CSV}")

    # -----------------------
    # Load features + F1 table
    # -----------------------
    df_feat = pd.read_excel(FEATURES_XLSX)
    df_feat.columns = [str(c).strip() for c in df_feat.columns]

    df_f1 = pd.read_csv(F1_ALL_MODELS_CSV, sep=None, engine="python")
    df_f1.columns = [str(c).strip() for c in df_f1.columns]

    if "trg_lang" not in df_f1.columns:
        raise ValueError(f"Expected 'trg_lang' in {F1_ALL_MODELS_CSV}. Got: {list(df_f1.columns)}")
    if f1_col is None or f1_col not in df_f1.columns:
        raise ValueError(
            f"Expected F1 column '{f1_col}' for EMB_MODEL='{EMB_MODEL}' in {F1_ALL_MODELS_CSV}.\n"
            f"Available columns: {list(df_f1.columns)}\n\n"
            f"Fix: update MODEL_SPECS['{EMB_MODEL}']['f1_col'] to match your CSV."
        )

    df_f1 = df_f1.rename(columns={"trg_lang": "lang", f1_col: "f1"})
    df_f1["lang"] = safe_strip_series(df_f1["lang"])
    df_f1["f1"] = df_f1["f1"].apply(parse_numeric_or_percent)

    if "lang" not in df_feat.columns:
        raise ValueError(f"Excel must contain a 'lang' column. Got: {list(df_feat.columns)}")

    df_feat["lang"] = safe_strip_series(df_feat["lang"])

    # -----------------------
    # Filter to overlap (Excel ∩ F1)
    # -----------------------
    excel_langs = set(df_feat["lang"].tolist())
    f1_langs = set(df_f1["lang"].tolist())
    common_langs = sorted(excel_langs & f1_langs)

    df_all = df_feat.merge(df_f1[["lang", "f1"]], on="lang", how="inner")
    df_all = df_all.dropna(subset=["f1"]).copy()

    print(f"Excel langs: {len(excel_langs)}")
    print(f"F1 langs   : {len(f1_langs)}")
    print(f"Overlap    : {len(common_langs)}")
    print(f"Usable rows after merge (non-NaN f1): {len(df_all)}\n")

    if len(df_all) < max(12, N_TEST + 6):
        raise ValueError(
            f"Too few usable rows after merge for N_TEST={N_TEST}.\n"
            f"Usable rows: {len(df_all)}. Increase overlap or reduce N_TEST."
        )

    # -----------------------
    # Random test split (choose 6 from the merged rows)
    # -----------------------
    rng = np.random.default_rng(RANDOM_STATE)
    all_langs = df_all["lang"].tolist()

    test_langs = sorted(rng.choice(all_langs, size=N_TEST, replace=False).tolist())
    df_test = df_all[df_all["lang"].isin(test_langs)].copy()
    df_train = df_all[~df_all["lang"].isin(test_langs)].copy()

    print("Chosen TEST_LANGS:", test_langs)
    print("Train shape:", df_train.shape)
    print("Test shape :", df_test.shape)

    # -----------------------
    # Build feature columns
    # -----------------------
    ID_COLS = {"lang", "language", "iso639_3", "f1"}

    # Keep ONLY the dist column for this model (if it exists); drop other dist_*
    drop_dist_cols = [
        c for c in df_all.columns
        if str(c).startswith("dist_") and (dist_col_keep is None or c != dist_col_keep)
    ]

    # Keep ONLY the training flag for this model (if it exists); drop other is_training_lang_*
    drop_training_flags = [
        c for c in df_all.columns
        if str(c).startswith("is_training_lang_") and (train_flag_keep is None or c != train_flag_keep)
    ]

    # TTR: keep whitespace + optional model-specific if present; drop other TTR_*
    keep_ttr_cols = {"TTR_whitspace", "TTR_whitespace"}  # both spellings safe
    if ttr_model_col is not None and ttr_model_col in df_all.columns:
        keep_ttr_cols.add(ttr_model_col)

    drop_ttr_cols = [
        c for c in df_all.columns
        if str(c).startswith("TTR_") and c not in keep_ttr_cols
    ]

    drop_cols = set(drop_dist_cols) | set(drop_training_flags) | set(drop_ttr_cols)

    # Optional: If your Excel has glot500_size but you're not using glot500, drop it
    if "glot500_size" in df_all.columns and EMB_MODEL != "glot500":
        drop_cols.add("glot500_size")

    feature_cols = [c for c in df_all.columns if c not in ID_COLS and c not in drop_cols]

    X_train = df_train[feature_cols].copy()
    y_train = df_train["f1"].astype(float).copy()

    X_test = df_test[feature_cols].copy()
    y_test = df_test["f1"].astype(float).copy()

    # -----------------------
    # Numeric normalization (robust)
    # -----------------------
    for col in X_train.columns:
        if X_train[col].dtype == "object":
            parsed = X_train[col].apply(parse_numeric_or_percent)
            ratio_numeric = np.isfinite(parsed.to_numpy(dtype=float)).mean()
            if ratio_numeric >= 0.70:
                X_train[col] = parsed
                X_test[col] = X_test[col].apply(parse_numeric_or_percent)

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    print(f"\nUsing {len(num_cols)} numeric + {len(cat_cols)} categorical columns")
    print(f"Total raw features before one-hot: {X_train.shape[1]}")

    # -----------------------
    # Preprocess pipeline
    # -----------------------
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            min_frequency=4,
            handle_unknown="infrequent_if_exist"
        )),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    # -----------------------
    # ElasticNetCV
    # -----------------------
    def fit_enet(l1_ratios: List[float]) -> Pipeline:
        enet = ElasticNetCV(
            l1_ratio=l1_ratios,
            cv=CV_FOLDS,
            random_state=RANDOM_STATE,
            max_iter=30000,
        )
        pipe = Pipeline([
            ("prep", preprocess),
            ("model", enet),
        ])
        pipe.fit(X_train, y_train)
        return pipe

    pipe = fit_enet([0.7, 0.85, 0.9, 0.95, 1.0])

    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_
    nonzero_idx = np.where(np.abs(coefs) > 1e-8)[0]
    n_selected = int(len(nonzero_idx))

    if n_selected > max(20, TARGET_SELECTED_FEATURES * 2):
        print(f"\n[auto-tighten] selected={n_selected} too many → refit with stronger sparsity")
        pipe = fit_enet([0.95, 1.0])
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pipe.named_steps["model"].coef_
        nonzero_idx = np.where(np.abs(coefs) > 1e-8)[0]
        n_selected = int(len(nonzero_idx))

    if n_selected > max(30, TARGET_SELECTED_FEATURES * 3):
        print(f"\n[auto-tighten] selected={n_selected} still high → refit with pure Lasso (l1_ratio=1.0)")
        pipe = fit_enet([1.0])
        feature_names = pipe.named_steps["prep"].get_feature_names_out()
        coefs = pipe.named_steps["model"].coef_
        nonzero_idx = np.where(np.abs(coefs) > 1e-8)[0]
        n_selected = int(len(nonzero_idx))

    model = pipe.named_steps["model"]
    intercept = float(model.intercept_)

    # -----------------------
    # Predict test langs
    # -----------------------
    pred = pipe.predict(X_test)
    if CLIP_TO_UNIT:
        pred = np.clip(pred, 0.0, 1.0)

    out = pd.DataFrame({
        "lang": df_test["lang"].values,
        "language": df_test.get("language", df_test["lang"]).values,
        "pred_f1": pred,
    })

    # Include kept dist column in outputs (if exists)
    if dist_col_keep is not None and dist_col_keep in df_test.columns:
        out[dist_col_keep] = df_test[dist_col_keep].apply(parse_numeric_or_percent).values

    out.to_csv(pred_csv, index=False, encoding="utf-8")
    print("\nSaved predictions to:", pred_csv)

    # -----------------------
    # Pred vs True + diff
    # -----------------------
    pred_vs_true = out.copy()
    pred_vs_true["true_f1"] = y_test.values
    pred_vs_true["diff"] = pred_vs_true["pred_f1"] - pred_vs_true["true_f1"]
    pred_vs_true = pred_vs_true.round({"pred_f1": 3, "true_f1": 3, "diff": 3})

    pred_vs_true.to_csv(pred_vs_true_csv, index=False, encoding="utf-8")
    print("Saved pred-vs-true table to:", pred_vs_true_csv)

    print("\n=== Predicted vs True (random test) ===")
    print(pred_vs_true.to_string(index=False))

    # PNG table
    if MAKE_PNG:
        # make a PNG WITHOUT any dist_* column (or any extra cols)
        png_cols = ["lang", "pred_f1", "true_f1", "diff"]
        png_tbl = pred_vs_true[png_cols].copy()

        fig, ax = plt.subplots(figsize=(9, 0.6 + 0.5 * len(png_tbl)))
        ax.axis("off")

        tbl_str = png_tbl.copy()
        for c in ["pred_f1", "true_f1", "diff"]:
            tbl_str[c] = tbl_str[c].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")

        table = ax.table(
            cellText=tbl_str.values,
            colLabels=tbl_str.columns,
            cellLoc="center",
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.35)

        fig.tight_layout()
        fig.savefig(compare_png, dpi=200)
        plt.close(fig)
        print("Saved PNG comparison table to:", compare_png)

    # -----------------------
    # Coefs
    # -----------------------
    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    coef_df.to_csv(coef_all_csv, index=False, encoding="utf-8")
    print("Saved ALL coefficients to:", coef_all_csv)

    selected_df = coef_df.loc[coef_df["abs_coef"] > 1e-8].copy()
    selected_df = selected_df.sort_values("abs_coef", ascending=False)
    selected_df.to_csv(coef_selected_csv, index=False, encoding="utf-8")
    print("Saved SELECTED coefficients to:", coef_selected_csv)

    # -----------------------
    # CV Score (train only)
    # -----------------------
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")
    neg_rmse_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")

    r2_mean = float(np.mean(r2_scores))
    r2_std = float(np.std(r2_scores))
    rmse_mean = float(-np.mean(neg_rmse_scores))
    rmse_std = float(np.std(-neg_rmse_scores))

    # -----------------------
    # Save report TXT
    # -----------------------
    lines: List[str] = []
    lines.append("=== ElasticNetCV Regression Report ===")
    lines.append(f"EMB_MODEL        : {EMB_MODEL}")
    lines.append(f"F1 column        : {f1_col}")
    lines.append(f"N_TEST           : {N_TEST}")
    lines.append(f"RANDOM_STATE     : {RANDOM_STATE}")
    lines.append("")
    lines.append(f"Total merged rows (usable): {len(df_all)}")
    lines.append(f"Train rows       : {len(df_train)}")
    lines.append(f"Test rows        : {len(df_test)}")
    lines.append(f"Test langs       : {', '.join(test_langs)}")
    lines.append("")
    lines.append(f"Chosen alpha     : {model.alpha_}")
    lines.append(f"Chosen l1_ratio  : {model.l1_ratio_}")
    lines.append(f"Intercept        : {intercept}")
    lines.append(f"Selected features: {n_selected}")
    lines.append("")
    lines.append(f"CV R² mean±std   : {r2_mean:.4f} ± {r2_std:.4f}")
    lines.append(f"CV RMSE mean±std : {rmse_mean:.4f} ± {rmse_std:.4f}")
    lines.append("")
    lines.append("Top selected features:")
    for _, row in selected_df.head(30).iterrows():
        lines.append(f"  {row['coef']:+.6f}  {row['feature']}")

    report_txt.write_text("\n".join(lines), encoding="utf-8")
    print("Saved model report to:", report_txt)

    print("\n============================================================")
    print("DONE ✅")
    print("------------------------------------------------------------")
    print("Predictions file  :", pred_csv)
    print("Pred-vs-true file :", pred_vs_true_csv)
    if MAKE_PNG:
        print("Compare PNG       :", compare_png)
    print("Selected coef file:", coef_selected_csv)
    print("Report file       :", report_txt)
    print("------------------------------------------------------------")
    print("\nSelected features (top 15):")
    print(selected_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
