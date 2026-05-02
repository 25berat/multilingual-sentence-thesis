#!/usr/bin/env python3
import argparse
import os
from glob import glob

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Optional, List
from matplotlib import patheffects as pe
from typing import List

# Optional: remove seaborn dependency; Matplotlib handles colors fine.
# If you really want seaborn palettes, uncomment:
# import seaborn as sns

# --- your post-processing (keep as-is) ---
from post_processing import cluster_based, whitening


# ------------- helpers -------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def guess_label_from_path(path: str) -> str:
    """
    ../embs/bucc/xlm-roberta-base/7/azj_Latn/emb.pt -> azj_Latn
    """
    return os.path.basename(os.path.dirname(path))


def colour_cycle(n: int):
    """
    Return a list of n distinct colours. Pure Matplotlib tab10/tab20 fallback.
    """
    # Prefer tab20 if we need many colours
    cmap = plt.get_cmap('tab20' if n > 10 else 'tab10')
    return [cmap(i % cmap.N) for i in range(n)]


def load_embs(emb_file: str, load: str, do_cbie: bool, do_whiten: bool):
    if load == "torch":
        # Support both Tensor and dict with "emb" key.
        obj = torch.load(emb_file, map_location="cpu")
        if isinstance(obj, dict):
            # common keys to try
            for k in ("emb", "embs", "embeddings"):
                if k in obj:
                    obj = obj[k]
                    break
        if hasattr(obj, "numpy"):
            embs = obj.numpy()
        else:
            embs = np.asarray(obj)
    elif load == "np":
        embs = np.load(emb_file, allow_pickle=True)
        # Handle .npz with "arr_0" / "emb" keys
        if isinstance(embs, np.lib.npyio.NpzFile):
            for k in ("emb", "embs", "embeddings", "arr_0"):
                if k in embs.files:
                    embs = embs[k]
                    break
            embs = np.asarray(embs)
    else:
        raise ValueError(f"Unknown load mode: {load}")

    if do_whiten:
        embs = whitening(embs)
    if do_cbie:
        embs = cluster_based(embs, n_cluster=7, n_pc=12, hidden_size=embs.shape[1])

    # Ensure 2D (num_items, hidden_size)
    embs = np.asarray(embs)
    if embs.ndim == 1:
        embs = embs.reshape(1, -1)
    return embs


def run_pca_tsne(all_embs: np.ndarray, pca_components: int, tsne_perplexity: float, tsne_seed: int):
    pca_components = min(pca_components, all_embs.shape[1])
    all_embs = PCA(n_components=pca_components).fit_transform(all_embs)
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=tsne_seed, init="pca")
    return tsne.fit_transform(all_embs)


# ------------- plotting modes -------------
def visualise_tsne_single(embs: np.ndarray, plot_file: str, title: str,
                          pca_components: int, perplexity: float, seed: int,
                          label: Optional[str] = None):
    embs_2d = run_pca_tsne(embs, pca_components, perplexity, seed)
    fig = plt.figure(figsize=(16, 12))
    plt.title(title)
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], s=10, label=label)
    if label:
        plt.legend(title="Language")
    ensure_dir(plot_file)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualise_tsne_pair(embs1: np.ndarray, embs2: np.ndarray, plot_file: str, title: str,
                        pca_components: int, perplexity: float, seed: int,
                        label1: str = "set1", label2: str = "set2"):
    all_embs = np.concatenate([embs1, embs2], axis=0)
    embs_2d = run_pca_tsne(all_embs, pca_components, perplexity, seed)

    n1 = len(embs1)
    fig = plt.figure(figsize=(16, 12))
    plt.title(title)
    cols = colour_cycle(2)
    plt.scatter(embs_2d[:n1, 0], embs_2d[:n1, 1], s=10, label=label1, color=cols[0])
    plt.scatter(embs_2d[n1:, 0], embs_2d[n1:, 1], s=10, label=label2, color=cols[1])
    plt.legend(title="Language")
    ensure_dir(plot_file)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualise_tsne_multi(
    emb_groups: List[np.ndarray],
    labels: List[str],
    plot_file: str,
    title: str,
    pca_components: int,
    perplexity: float,
    seed: int,
    overlay_centroids: bool = True,
    enumerate_labels: bool = True,
):
    """
    Plot multiple embedding sets (one per language) in a single t-SNE.
    If enumerate_labels=True, place a number at each language centroid
    and draw a right-side mapping "number -> language".
    """
    # sizes and joint t-SNE
    sizes = [e.shape[0] for e in emb_groups]
    all_embs = np.concatenate(emb_groups, axis=0)
    embs_2d = run_pca_tsne(all_embs, pca_components, perplexity, seed)

    # split back to per-language slices
    idxs = np.cumsum([0] + sizes)
    cols = colour_cycle(len(emb_groups))

    fig = plt.figure(figsize=(16, 12))
    ax = plt.gca()
    plt.title(title)

    # draw points per language
    centroids_xy = []
    for i, (start, end) in enumerate(zip(idxs[:-1], idxs[1:])):
        xy = embs_2d[start:end]
        ax.scatter(xy[:, 0], xy[:, 1], s=10, label=labels[i], color=cols[i], alpha=0.85)
        if overlay_centroids:
            cxy = xy.mean(axis=0)
            centroids_xy.append((cxy[0], cxy[1], labels[i], cols[i]))

    # optional: overlay big centroid markers + label
    if overlay_centroids and centroids_xy:
        for x, y, lab, col in centroids_xy:
            ax.scatter([x], [y], s=160, marker='X', color=col, edgecolors='k', linewidths=0.8, zorder=5)
            ax.text(x + 1.0, y + 1.0, lab, fontsize=9, weight='bold')

    # optional: enumerate each cluster at its centroid and print mapping
    if enumerate_labels:
        mapping_lines = []
        for i, (start, end) in enumerate(zip(idxs[:-1], idxs[1:])):
            xy = embs_2d[start:end]
            cx, cy = xy.mean(axis=0)
            txt = ax.text(
                cx, cy, str(i + 1),
                fontsize=12, weight='bold', ha='center', va='center', color='white', zorder=6
            )
            # outline for readability on any background
            txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground='black')])
            mapping_lines.append(f"{i+1}. {labels[i]}")
        # right-side legend box
        fig.text(
            0.995, 0.5, "\n".join(mapping_lines),
            ha='right', va='center', fontsize=9, family='monospace',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85)
        )

    ax.legend(title="Language", loc="best", frameon=True)
    ensure_dir(plot_file)
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close(fig)



# ------------- CLI -------------
def main(args):
    # If user supplied --emb_files OR --emb_dir we do multi-mode. Otherwise single/parallel.
    if args.emb_dir:
        # Collect emb files from a folder (non-recursive or recursive)
        pattern = os.path.join(args.emb_dir, "**" if args.recursive else "*", args.emb_name)
        files = sorted(glob(pattern, recursive=args.recursive))
        if not files:
            raise FileNotFoundError(f"No files match: {pattern}")
        args.emb_files = files

    if args.emb_files:
        # Multi-language plot
        emb_groups, labels = [], []
        for f in args.emb_files:
            emb_groups.append(load_embs(f, args.load, args.do_cbie, args.do_whiten))
            labels.append(guess_label_from_path(f))
        if args.labels:
            if len(args.labels) != len(labels):
                raise ValueError("--labels length must match --emb_files length.")
            labels = args.labels

        plot_file = args.plot_file or f"../plots/{args.dataset}/{args.model}/{args.layer}/MULTI/tsne{args.append_file_name}.png"
        title = args.title or f"t-SNE (multi) of {len(emb_groups)} sets | whiten={args.do_whiten} cbie={args.do_cbie}"
        visualise_tsne_multi(
            emb_groups, labels, plot_file, title,
            pca_components=args.pca_components,
            perplexity=args.perplexity,
            seed=args.seed,
            overlay_centroids=args.overlay_centroids,
            enumerate_labels=args.enumerate
        )
        return

    # Single or pair
    if not args.emb_file:
        lang = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng1'
        args.emb_file = f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang}{args.append_file_name}.pt'

    embs = load_embs(args.emb_file, args.load, args.do_cbie, args.do_whiten)

    if args.parallel_vis:
        if not args.parallel_emb_file:
            lang = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng2'
            args.parallel_emb_file = f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang}{args.append_file_name}.pt'
        parallel = load_embs(args.parallel_emb_file, args.load, args.do_cbie, args.do_whiten)

        plot_file = args.plot_file or f'../plots/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/PAIR_tsne{args.append_file_name}.png'
        l1 = guess_label_from_path(args.emb_file)
        l2 = guess_label_from_path(args.parallel_emb_file)
        title = args.title or f"t-SNE: {l1} vs {l2} | whiten={args.do_whiten} cbie={args.do_cbie}"
        visualise_tsne_pair(
            embs, parallel, plot_file, title,
            pca_components=args.pca_components,
            perplexity=args.perplexity,
            seed=args.seed,
            label1=l1, label2=l2
        )
    else:
        plot_file = args.plot_file or f'../plots/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/tsne{args.append_file_name}.png'
        label = guess_label_from_path(args.emb_file)
        title = args.title or f"t-SNE: {label} | whiten={args.do_whiten} cbie={args.do_cbie}"
        visualise_tsne_single(
            embs, plot_file, title,
            pca_components=args.pca_components,
            perplexity=args.perplexity,
            seed=args.seed,
            label=label
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='t-SNE visualisation of embeddings (single / pair / multi).')

    # Original path pieces (still supported)
    parser.add_argument('--model', type=str, help="model name for path building (legacy mode)")
    parser.add_argument('--layer', type=int, help="layer id for path building (legacy mode)")
    parser.add_argument('--dataset', type=str, default="tatoeba", choices=["tatoeba", "wiki", "sts", "bucc"])
    parser.add_argument('--append_file_name', type=str, default="", help='e.g., _whitened')
    parser.add_argument('--lang_or_track', type=str, default="", help='path component')

    # Direct inputs
    parser.add_argument('--emb_file', type=str, help="single file to plot (legacy single mode)")
    parser.add_argument('--parallel_emb_file', type=str, help="second file (legacy pair mode)")
    parser.add_argument('--plot_file', type=str, help="output plot path")

    # Multi-mode
    parser.add_argument('--emb_files', nargs='+', help='list of embedding files to plot together')
    parser.add_argument('--labels', nargs='+', help='legend labels for emb_files (must match length)')
    parser.add_argument('--emb_dir', type=str, help='folder that contains many subfolders with an embedding file')
    parser.add_argument('--emb_name', type=str, default='emb.pt', help='embedding file name to look for inside each subfolder')
    parser.add_argument('--recursive', action='store_true', help='search for emb_name recursively inside emb_dir')

    # Alg / transforms
    parser.add_argument('--parallel_vis', action='store_true', help='legacy two-file mode')
    parser.add_argument('--load', type=str, default='torch', choices=["torch", "np"])
    parser.add_argument('--do_cbie', action='store_true')
    parser.add_argument('--do_whiten', action='store_true')

    # t-SNE / PCA controls
    parser.add_argument('--pca_components', type=int, default=50)
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=0)

    # Cosmetics / extras
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--overlay_centroids', action='store_true',
                        help='draw one centroid per language on the multi plot')
    parser.add_argument('--enumerate', action='store_true',
                        help='place numbers at cluster centroids and add a number→language legend')

    args = parser.parse_args()

    # If user supplied --emb_files OR --emb_dir we do multi-mode. Otherwise single/parallel.
    if args.emb_dir:
        from glob import glob
        pattern = os.path.join(args.emb_dir, "**" if args.recursive else "*", args.emb_name)
        files = sorted(glob(pattern, recursive=args.recursive))
        if not files:
            raise FileNotFoundError(f"No files match: {pattern}")
        args.emb_files = files

    if args.emb_files:
        # Multi-language plot
        emb_groups, labels = [], []
        for f in args.emb_files:
            emb_groups.append(load_embs(f, args.load, args.do_cbie, args.do_whiten))
            labels.append(guess_label_from_path(f))
        if args.labels:
            if len(args.labels) != len(labels):
                raise ValueError("--labels length must match --emb_files length.")
            labels = args.labels

        plot_file = args.plot_file or f"../plots/{args.dataset}/{args.model}/{args.layer}/MULTI/tsne{args.append_file_name}.png"
        title = args.title or f"t-SNE (multi) of {len(emb_groups)} sets | whiten={args.do_whiten} cbie={args.do_cbie}"
        visualise_tsne_multi(
            emb_groups, labels, plot_file, title,
            pca_components=args.pca_components,
            perplexity=args.perplexity,
            seed=args.seed,
            overlay_centroids=args.overlay_centroids,
            enumerate_labels=args.enumerate
        )
    else:
        # Single or pair modes (legacy)
        if not args.emb_file:
            lang = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng1'
            args.emb_file = f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang}{args.append_file_name}.pt'

        embs = load_embs(args.emb_file, args.load, args.do_cbie, args.do_whiten)

        if args.parallel_vis:
            if not args.parallel_emb_file:
                lang = args.lang_or_track if args.dataset in ['tatoeba', 'wiki'] else 'lng2'
                args.parallel_emb_file = f'../embs/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/{lang}{args.append_file_name}.pt'
            parallel = load_embs(args.parallel_emb_file, args.load, args.do_cbie, args.do_whiten)

            plot_file = args.plot_file or f'../plots/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/PAIR_tsne{args.append_file_name}.png'
            l1 = guess_label_from_path(args.emb_file)
            l2 = guess_label_from_path(args.parallel_emb_file)
            title = args.title or f"t-SNE: {l1} vs {l2} | whiten={args.do_whiten} cbie={args.do_cbie}"
            visualise_tsne_pair(
                embs, parallel, plot_file, title,
                pca_components=args.pca_components,
                perplexity=args.perplexity,
                seed=args.seed,
                label1=l1, label2=l2
            )
        else:
            plot_file = args.plot_file or f'../plots/{args.dataset}/{args.model}/{args.layer}/{args.lang_or_track}/tsne{args.append_file_name}.png'
            label = guess_label_from_path(args.emb_file)
            title = args.title or f"t-SNE: {label} | whiten={args.do_whiten} cbie={args.do_cbie}"
            visualise_tsne_single(
                embs, plot_file, title,
                pca_components=args.pca_components,
                perplexity=args.perplexity,
                seed=args.seed,
                label=label
            )

