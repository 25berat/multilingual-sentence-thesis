#!/usr/bin/env python3
"""
Distance analysis helper.

Input:
  - dist_from_deu_Latn_avg.tsv
  - dist_from_tur_Latn_avg.tsv

Outputs (written to OUT_DIR):
  1) mean_distance_by_model__<ref>.tsv
  2) top5_bottom5_by_model__<ref>.tsv
  3) top5_bottom5_by_avg__<ref>.tsv   (uses avg_distance column)

Notes:
- Excludes the reference language row where avg_distance == 0 (e.g., tur_Latn in Turkish file) automatically.
- Keeps numeric precision as-is; you can change float_format if you want rounding.
"""

from pathlib import Path
import pandas as pd


DEU_PATH = Path(r"C:\Users\berat\PycharmProjects\outliers\src\centroids\raw\dist_from_deu_Latn_avg.tsv")
TUR_PATH = Path(r"C:\Users\berat\PycharmProjects\outliers\src\centroids\raw\dist_from_tur_Latn_avg.tsv")
OUT_DIR  = Path(r"C:\Users\berat\PycharmProjects\outliers\src\centroids\raw\distance_analysis")


# -----------------------------
# helpers
# -----------------------------
def read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # convert numeric columns to float (everything except "lang")
    for c in df.columns:
        if c != "lang":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def drop_reference_row(df: pd.DataFrame) -> pd.DataFrame:
    # drop rows where avg_distance is exactly 0 (reference language)
    if "avg_distance" in df.columns:
        return df.loc[~(df["avg_distance"].fillna(1.0) == 0.0)].copy()
    return df.copy()


def model_columns(df: pd.DataFrame) -> list[str]:
    # model columns are all numeric columns except avg_distance
    cols = [c for c in df.columns if c not in ("lang", "avg_distance")]
    return cols


def mean_by_model(df: pd.DataFrame) -> pd.DataFrame:
    cols = model_columns(df)
    means = df[cols].mean(axis=0, skipna=True).sort_values()
    out = means.reset_index()
    out.columns = ["model", "mean_distance"]
    return out


def top_bottom_by_avg(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    if "avg_distance" not in df.columns:
        raise ValueError("Expected an 'avg_distance' column.")
    tmp = df[["lang", "avg_distance"]].dropna().sort_values("avg_distance", ascending=True)
    top = tmp.head(n).assign(rank_group="closest_avg", rank=range(1, n + 1))
    bottom = tmp.tail(n).sort_values("avg_distance", ascending=False).assign(rank_group="farthest_avg", rank=range(1, n + 1))
    return pd.concat([top, bottom], ignore_index=True)[["rank_group", "rank", "lang", "avg_distance"]]


def top_bottom_by_model(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    cols = model_columns(df)
    rows = []
    for m in cols:
        tmp = df[["lang", m]].dropna().sort_values(m, ascending=True)
        top = tmp.head(n)
        bottom = tmp.tail(n).sort_values(m, ascending=False)

        for i, r in enumerate(top.itertuples(index=False), start=1):
            rows.append({"model": m, "rank_group": "closest", "rank": i, "lang": r.lang, "distance": getattr(r, m)})
        for i, r in enumerate(bottom.itertuples(index=False), start=1):
            rows.append({"model": m, "rank_group": "farthest", "rank": i, "lang": r.lang, "distance": getattr(r, m)})

    out = pd.DataFrame(rows)
    # nice ordering: model, closest then farthest
    out["rank_group"] = pd.Categorical(out["rank_group"], categories=["closest", "farthest"], ordered=True)
    out = out.sort_values(["model", "rank_group", "rank"])
    return out


def save_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def run_one(ref_name: str, in_path: Path) -> None:
    df = read_tsv(in_path)
    df = drop_reference_row(df)

    # 1) model-wise mean distances (across languages)
    mean_df = mean_by_model(df)
    save_tsv(mean_df, OUT_DIR / f"mean_distance_by_model__{ref_name}.tsv")

    # 2) top/bottom 5 by avg_distance
    avg_tb = top_bottom_by_avg(df, n=5)
    save_tsv(avg_tb, OUT_DIR / f"top5_bottom5_by_avg__{ref_name}.tsv")

    # 3) top/bottom 5 by each model column
    model_tb = top_bottom_by_model(df, n=5)
    save_tsv(model_tb, OUT_DIR / f"top5_bottom5_by_model__{ref_name}.tsv")

    print(f"[OK] Wrote outputs for {ref_name} to: {OUT_DIR}")


def main():
    run_one("deu_Latn", DEU_PATH)
    run_one("tur_Latn", TUR_PATH)


if __name__ == "__main__":
    main()