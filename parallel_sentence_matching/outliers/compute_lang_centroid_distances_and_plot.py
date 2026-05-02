#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import argparse
from pathlib import Path   # <-- NEU
import numpy as np
import matplotlib.pyplot as plt
...
# models we try automatically (when aggregating)
MODELS = ["glot500", "labse", "sonar", "laser", "sonar", "qwen3", "llama31"]

# ---- FIX: anchor BASE_DIR to this script's src folder ----
HERE = Path(__file__).resolve()      # .../outliers/src/compute_lang_....
SRC_DIR = HERE.parent                # .../outliers/src
BASE_DIR = SRC_DIR / "centroids" /"raw"     # .../outliers/src/centroids

REF_LANG_DEFAULT = "tur_Latn"


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def load_centroids(npz_path: str):
    data = np.load(npz_path)
    # deterministic order
    langs = sorted(data.files)
    centroids = np.stack([data[l] for l in langs])
    return langs, centroids


def compute_distance_matrix(C: np.ndarray, langs: list, metric: str, out_csv: str):
    D = cdist(C, C, metric=metric)
    ensure_dir(out_csv)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("lang," + ",".join(langs) + "\n")
        for i, li in enumerate(langs):
            f.write(f"{li}," + ",".join(f"{D[i, j]:.6f}" for j in range(len(langs))) + "\n")
    print(f"[✓] Distance matrix saved to {out_csv}")
    return D


def plot_tsne(C: np.ndarray, langs: list, out_png: str, title: str):
    # reduce to max 50 dims before t-SNE
    reduced = PCA(n_components=min(50, C.shape[0], C.shape[1])).fit_transform(C)
    tsne = TSNE(
        n_components=2,
        random_state=0,
        perplexity=min(10, len(langs) - 1),
        init="pca",
    )
    X2d = tsne.fit_transform(reduced)

    plt.figure(figsize=(14, 10))
    plt.scatter(X2d[:, 0], X2d[:, 1], s=60)
    for x, y, l in zip(X2d[:, 0], X2d[:, 1], langs):
        plt.text(x + 0.5, y + 0.5, l, fontsize=8)
    plt.title(title)
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] t-SNE plot saved to {out_png}")


def save_heatmap(df_vals: np.ndarray, langs: list, out_png: str, title: str):
    plt.figure(figsize=(14, 12))
    im = plt.imshow(df_vals, cmap="viridis")
    plt.colorbar(im, label="Distance")
    plt.xticks(range(len(langs)), langs, rotation=45, ha="right")
    plt.yticks(range(len(langs)), langs)
    plt.title(title)
    plt.tight_layout()
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[✓] Heatmap saved to {out_png}")


def save_table(df_vals: np.ndarray, langs: list, out_png: str, title: str):
    vals = np.round(df_vals, 4)
    fig, ax = plt.subplots(figsize=(len(langs) * 0.7, len(langs) * 0.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=vals,
        rowLabels=langs,
        colLabels=langs,
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
    ensure_dir(out_png)
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[✓] Table image saved to {out_png}")


def save_reflang_views(ref_lang: str, langs: list, D: np.ndarray, out_dir: str, metric: str):
    """
    Create 2 extra views for a reference language for THIS model:
    1) PNG table: language vs distance-from-ref
    2) Horizontal bar chart: sorted by distance
    Also return a dict {lang: distance} so we can aggregate later.
    """
    dist_dict = {}

    if ref_lang not in langs:
        print(f"[warn] reference language {ref_lang} not found in centroids, skipping…")
        return dist_dict

    ref_idx = langs.index(ref_lang)
    distances = D[ref_idx, :]  # shape (N,)

    # fill dict for later averaging
    for lang, d in zip(langs, distances):
        dist_dict[lang] = float(d)

    # sort ascending for nicer plots
    sorted_idx = np.argsort(distances)
    sorted_langs = [langs[i] for i in sorted_idx]
    sorted_dists = distances[sorted_idx]

    # 1) table png
    fig, ax = plt.subplots(figsize=(6, max(3, len(sorted_langs) * 0.25)))
    ax.axis("off")
    cell_text = [[l, f"{d:.6f}"] for l, d in zip(sorted_langs, sorted_dists)]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=["lang", f"dist_from_{ref_lang} ({metric})"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.1)
    out_table = os.path.join(out_dir, f"dist_from_{ref_lang}_table.png")
    plt.title(f"Distances from {ref_lang}")
    plt.tight_layout()
    plt.savefig(out_table, dpi=300)
    plt.close()
    print(f"[✓] saved ref-lang table → {out_table}")

    # 2) bar chart
    fig, ax = plt.subplots(figsize=(10, max(3, len(sorted_langs) * 0.25)))
    ax.barh(sorted_langs, sorted_dists)
    ax.set_xlabel(f"distance ({metric})")
    ax.set_title(f"Distances from {ref_lang}")
    plt.tight_layout()
    out_bar = os.path.join(out_dir, f"dist_from_{ref_lang}_bar.png")
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print(f"[✓] saved ref-lang bar → {out_bar}")

    return dist_dict


def write_avg_ref_dist(ref_lang: str, per_model_dicts: dict, out_path: str, title_suffix: str = ""):
    """
    per_model_dicts:
        {
          "glot500": {lang: dist, ...},
          "labse":   {lang: dist, ...},
          ...
        }
    We create a union of all langs, then average over models that have that lang.
    Also: create a bar chart + table PNG for the averaged values, with model columns.
    """
    # union of languages
    all_langs = set()
    for md in per_model_dicts.values():
        all_langs.update(md.keys())

    # fix model column order
    model_cols = list(per_model_dicts.keys())

    rows = []
    for lang in sorted(all_langs):
        model_vals = {}
        vals = []
        for model_name, md in per_model_dicts.items():
            v = md.get(lang)
            if v is not None:
                vals.append(v)
                model_vals[model_name] = v
            else:
                model_vals[model_name] = ""
        if lang == ref_lang:
            avg = 0.0
        else:
            avg = sum(vals) / len(vals) if vals else ""
        row = {"lang": lang, "avg_distance": avg}
        row.update(model_vals)
        rows.append(row)

    # write TSV
    ensure_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        header = ["lang", "avg_distance"] + model_cols
        f.write("\t".join(header) + "\n")
        for r in rows:
            line = [
                r["lang"],
                f"{r['avg_distance']:.6f}" if isinstance(r["avg_distance"], float) else "",
            ]
            for mc in model_cols:
                v = r.get(mc, "")
                if isinstance(v, float):
                    line.append(f"{v:.6f}")
                else:
                    line.append("")
            f.write("\t".join(line) + "\n")
    print(f"[✓] wrote average distances for {ref_lang} → {out_path}")

    # --- bar chart for avg ---
    # filter to those that actually have a numeric avg
    langs = []
    dists = []
    for r in rows:
        if isinstance(r["avg_distance"], float):
            langs.append(r["lang"])
            dists.append(r["avg_distance"])

    # sort by distance
    sort_idx = np.argsort(dists)
    langs_sorted = [langs[i] for i in sort_idx]
    dists_sorted = [dists[i] for i in sort_idx]

    # bar chart
    fig, ax = plt.subplots(figsize=(10, max(3, len(langs_sorted) * 0.25)))
    ax.barh(langs_sorted, dists_sorted)
    ax.set_xlabel("average distance (across models)")
    title = f"Average distance from {ref_lang}"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title)
    plt.tight_layout()
    out_bar = out_path.replace(".tsv", ".png").replace(".csv", ".png")
    plt.savefig(out_bar, dpi=300)
    plt.close()
    print(f"[✓] wrote avg bar chart → {out_bar}")

    # --- table PNG with model columns ---
    fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * 0.25)))
    ax.axis("off")

    # build cellText: each row = [lang, avg, glot500, labse, sonar]
    cell_text = []
    for r in rows:
        row_cells = [
            r["lang"],
            f"{r['avg_distance']:.6f}" if isinstance(r["avg_distance"], float) else "",
        ]
        for mc in model_cols:
            v = r.get(mc, "")
            if isinstance(v, float):
                row_cells.append(f"{v:.6f}")
            else:
                row_cells.append("")
        cell_text.append(row_cells)

    col_labels = ["lang", "avg_distance"] + model_cols
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.05)

    out_table = out_path.replace(".tsv", "_table.png").replace(".csv", "_table.png")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_table, dpi=300)
    plt.close()
    print(f"[✓] wrote avg table → {out_table}")


def infer_model_variant_from_path(centroids_npz: str):
    """
    Understand paths like:
      src/centroids/raw/glot500/centroids.npz
      src/centroids/cbie/labse/centroids.npz
      src/centroids/whitened/sonar/centroids.npz
    or old:
      src/centroids/glot500/centroids.npz
    """
    parts = os.path.normpath(centroids_npz).split(os.sep)
    # find "centroids"
    if "centroids" not in parts:
        return None, None
    idx = parts.index("centroids")
    # after "centroids" we may have variant + model
    model = None
    variant = None
    if len(parts) >= idx + 3:
        # centroids/<variant>/<model>/...
        variant = parts[idx + 1]
        model = parts[idx + 2]
    elif len(parts) >= idx + 2:
        # centroids/<model>/...
        model = parts[idx + 1]
    return model, variant


def process_one_centroids(npz_path: str, metric: str, ref_lang: str):
    """
    Process exactly one centroids.npz (this is the path that was passed in).
    Returns (model, variant, ref_dict)
    """
    model, variant = infer_model_variant_from_path(npz_path)
    if model is None:
        # fallback to old behavior
        # e.g. src/centroids/glot500/centroids.npz
        model = os.path.basename(os.path.dirname(npz_path))
    # build output dir
    if variant:
        out_dir = os.path.join(BASE_DIR, variant, model)
    else:
        out_dir = os.path.join(BASE_DIR, model)
    os.makedirs(out_dir, exist_ok=True)

    langs, C = load_centroids(npz_path)

    out_csv = os.path.join(out_dir, f"lang_distances_centroids_{model}.csv")
    D = compute_distance_matrix(C, langs, metric, out_csv)

    out_tsne = os.path.join(out_dir, f"lang_centroids_tsne_{model}.png")
    plot_tsne(C, langs, out_tsne, f"t-SNE of Language Centroids ({model}, {metric})")

    out_heat = os.path.join(out_dir, f"lang_distances_centroids_{model}_heatmap.png")
    save_heatmap(D, langs, out_heat, f"Language Distance Matrix ({model}, {metric})")

    out_table = os.path.join(out_dir, f"lang_distances_centroids_{model}_table.png")
    save_table(D, langs, out_table, f"Language Distance Matrix ({model}, {metric})")

    ref_dict = save_reflang_views(ref_lang, langs, D, out_dir, metric)
    return model, variant, ref_dict


def collect_and_write_variant_avg(variant: str, ref_lang: str, metric: str):
    """
    After we've processed one model for a variant, check if we can load the other two
    models of the same variant and write the avg file in src/centroids/<variant>/...
    """
    per_model = {}
    for m in MODELS:
        csv_path = os.path.join(BASE_DIR, variant, m, f"lang_distances_centroids_{m}.csv")
        if not os.path.isfile(csv_path):
            print(f"[warn] skipping {m}, not found: {csv_path}")
            continue

        # load the csv to get the row for ref_lang
        langs = []
        dists = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
            langs = header[1:]
            for line in f:
                parts = line.strip().split(",")
                lang = parts[0]
                vals = [float(x) for x in parts[1:]]
                if lang == ref_lang:
                    # this row: distances from ref_lang to all langs
                    dists = dict(zip(langs, vals))
                    break

        if not dists:
            print(f"[warn] {ref_lang} not in {csv_path}, skipping")
            continue

        per_model[m] = dists

    if not per_model:
        print("[!] no per-model reference distances collected, skipping avg TSV")
        return

    out_path = os.path.join(BASE_DIR, variant, f"dist_from_{ref_lang}_avg.tsv")
    write_avg_ref_dist(ref_lang, per_model, out_path, title_suffix=variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--centroids_npz", help="path to a centroids.npz (supports new variant layout)")
    ap.add_argument("--ref_lang", default=REF_LANG_DEFAULT)
    ap.add_argument("--metric", default="euclidean")
    args = ap.parse_args()

    if args.centroids_npz:
        # single-file mode (used by run_all_models.py)
        model, variant, ref_dict = process_one_centroids(
            args.centroids_npz, args.metric, args.ref_lang
        )
        if variant:
            collect_and_write_variant_avg(variant, args.ref_lang, args.metric)
        else:
            # old style: collect for flat layout
            # we mimic old behavior: try to read src/centroids/<model>/...
            per_model = {}
            if ref_dict:
                per_model[model] = ref_dict
            # also try the other models in flat layout
            for m in MODELS:
                if m == model:
                    continue
                other_csv = os.path.join(BASE_DIR, m, f"lang_distances_centroids_{m}.csv")
                if not os.path.isfile(other_csv):
                    continue
                # load row
                langs = []
                dists = {}
                with open(other_csv, "r", encoding="utf-8") as f:
                    header = f.readline().strip().split(",")
                    langs = header[1:]
                    for line in f:
                        parts = line.strip().split(",")
                        lang = parts[0]
                        vals = [float(x) for x in parts[1:]]
                        if lang == args.ref_lang:
                            dists = dict(zip(langs, vals))
                            break
                if dists:
                    per_model[m] = dists
            if per_model:
                avg_out = os.path.join(BASE_DIR, f"dist_from_{args.ref_lang}_avg.tsv")
                write_avg_ref_dist(args.ref_lang, per_model, avg_out)
        return

    # no arg → old behavior: process flat src/centroids/<model>/centroids.npz
    metric = args.metric
    ref_lang = args.ref_lang
    per_model_ref_dists = {}

    for model in MODELS:
        npz_path = os.path.join(BASE_DIR, model, "centroids.npz")
        if not os.path.isfile(npz_path):
            print(f"[warn] skipping {model}, not found: {npz_path}")
            continue

        _, _, ref_dict = process_one_centroids(npz_path, metric, ref_lang)
        if ref_dict:
            per_model_ref_dists[model] = ref_dict

    if per_model_ref_dists:
        avg_out = os.path.join(BASE_DIR, f"dist_from_{ref_lang}_avg.tsv")
        write_avg_ref_dist(ref_lang, per_model_ref_dists, avg_out)
    else:
        print("[!] no per-model reference distances collected, skipping avg TSV")


if __name__ == "__main__":
    main()
