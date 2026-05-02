#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"/centroids"
MODELS = ["glot500", "labse", "sonar", "laser", "sonar"]

# existing file from your run:
SELECTED_TSV = os.path.join(BASE_DIR, "sum_distances_selected_all_models.tsv")

# new file we will generate:
ALL_LANGS_TSV = os.path.join(BASE_DIR, "sum_distances_selected_all_models_all_languages.tsv")

# output pngs
OUT_SELECTED_PNG = os.path.join(BASE_DIR, "sum_distances_selected_all_models.png")
OUT_ALL_LANGS_PNG = os.path.join(BASE_DIR, "sum_distances_selected_all_models_all_languages.png")


def save_table_png(df: pd.DataFrame, out_path: str, title: str):
    df_round = df.round(6)

    fig_w = max(6, len(df_round.columns) * 1.0)
    fig_h = max(3, len(df_round.index) * 0.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=df_round.values,
        rowLabels=df_round.index,
        colLabels=df_round.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)

    for _, cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)

    plt.title(title, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] saved table → {out_path}")


def plot_existing_selected():
    """Plot the file you already have: sum_distances_selected_all_models.tsv"""
    if not os.path.isfile(SELECTED_TSV):
        raise FileNotFoundError(SELECTED_TSV)
    df = pd.read_csv(SELECTED_TSV, sep="\t")
    # index like: (model, lang)
    df_disp = df.set_index(["model", "lang"])
    save_table_png(df_disp, OUT_SELECTED_PNG, "Selected langs – all models")


def build_all_languages_tsv():
    """
    Read every src/centroids/<model>/sum_distances_all_<model>.tsv
    and outer-join them on 'lang' so we get a union of all langs.
    """
    frames = []
    for m in MODELS:
        path = os.path.join(BASE_DIR, m, f"sum_distances_all_{m}.tsv")
        if not os.path.isfile(path):
            print(f"[warn] missing {path}, skipping")
            continue
        df_m = pd.read_csv(path, sep="\t")
        # df_m: lang, sum_distance
        df_m = df_m.rename(columns={"sum_distance": m})
        frames.append(df_m)

    if not frames:
        raise RuntimeError("No per-model TSVs found, cannot build union.")

    # outer-join on lang
    merged = frames[0]
    for df_m in frames[1:]:
        merged = pd.merge(merged, df_m, on="lang", how="outer")

    # sort by lang for nicer display
    merged = merged.sort_values("lang").reset_index(drop=True)

    # save
    merged.to_csv(ALL_LANGS_TSV, sep="\t", index=False)
    print(f"[✓] wrote union TSV → {ALL_LANGS_TSV}")
    return merged


def plot_all_languages(merged: pd.DataFrame):
    # set index to lang so table looks nice
    df_disp = merged.set_index("lang")
    save_table_png(df_disp, OUT_ALL_LANGS_PNG, "All languages – all models")


def main():
    # 1) plot the existing selected file
    plot_existing_selected()

    # 2) build union and plot it
    merged = build_all_languages_tsv()
    plot_all_languages(merged)


if __name__ == "__main__":
    main()
