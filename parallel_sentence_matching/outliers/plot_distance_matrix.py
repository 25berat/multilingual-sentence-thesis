#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

# --- config ---
BASE_DIR = r"/centroids"
SELECTED_TSV = os.path.join(BASE_DIR, "sum_distances_selected_all_models.tsv")
MODELS = ["glot500", "labse", "sonar", "laser", "sonar"]   # adjust if you add more
OUT_SELECTED_PNG = os.path.join(BASE_DIR, "sum_distances_selected_all_models.png")
OUT_ALL_PNG = os.path.join(BASE_DIR, "sum_distances_all_models.png")


def save_table_png(df: pd.DataFrame, out_path: str, title: str):
    # round for readability
    df_round = df.round(6)

    # size: make it depend on shape a bit
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

    # add thin grid
    for _, cell in tbl.get_celld().items():
        cell.set_linewidth(0.5)

    plt.title(title, pad=20)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] saved table → {out_path}")


def part1_selected_table():
    if not os.path.isfile(SELECTED_TSV):
        raise FileNotFoundError(f"Not found: {SELECTED_TSV}")

    df = pd.read_csv(SELECTED_TSV, sep="\t")
    # typical columns: model | lang | sum_distance | rank | n_langs
    # make model+lang the index to look nice
    df_disp = df.set_index(["model", "lang"])
    save_table_png(df_disp, OUT_SELECTED_PNG, "Sum distances – selected langs (all models)")


def part2_all_models_table():
    """
    Read each model's
      src/centroids/{model}/sum_distances_all_{model}.tsv
    and combine into one big pivot:
        index = lang
        columns = model
        values = sum_distance
    """
    frames = []
    for m in MODELS:
        path = os.path.join(BASE_DIR, m, f"sum_distances_all_{m}.tsv")
        if not os.path.isfile(path):
            print(f"[warn] missing {path}, skipping")
            continue
        df_m = pd.read_csv(path, sep="\t")
        # expected columns: lang, sum_distance
        df_m["model"] = m
        frames.append(df_m)

    if not frames:
        print("[!] no model files found, aborting part 2")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # pivot to get: rows=lang, cols=model, values=sum_distance
    pivot = all_df.pivot(index="lang", columns="model", values="sum_distance")

    # sort rows alphabetically to be deterministic
    pivot = pivot.sort_index()

    save_table_png(pivot, OUT_ALL_PNG, "Sum distances – ALL langs × ALL models")


def main():
    part1_selected_table()
    part2_all_models_table()


if __name__ == "__main__":
    main()
