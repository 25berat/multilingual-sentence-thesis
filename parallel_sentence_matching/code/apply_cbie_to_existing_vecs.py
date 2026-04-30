#!/usr/bin/env python3
"""
Auto-CBIE for PaSeMiLL results.

- Detects all models under:
    <project_root>/results/<PIVOT_LANG>/raw/<backend>/

- Reads RAW embeddings from:
    .../raw/<backend>/embeddings/doc/**/*.vec

- Writes CBIE embeddings to:
    .../<PIVOT_LANG>/cbie/<backend>/embeddings/doc/**/*.vec

NEW behavior:
- By default, SKIPS a file if the CBIE output already exists.
- Use --overwrite to force recompute + overwrite.
- Optionally, use --recompute-if-stale to recompute only when RAW is newer than CBIE.

CBIE params:
    n_cluster = 7
    n_pc      = 12

NEW (your request):
- You can hardcode which backends to run via HARD_BACKENDS below.
  Example for only llama31_noprompt:
      HARD_BACKENDS = ["llama31_noprompt"]
  Example for llama31_noprompt + noprompt:
      HARD_BACKENDS = ["llama31_noprompt", "llama31_noprompt"]
  If HARD_BACKENDS = None, the script processes ALL backends under raw/.
"""

from pathlib import Path
import argparse
import numpy as np
from scipy import cluster as clst
from sklearn.decomposition import PCA

# =======================
# CHANGE THIS WHEN NEEDED
# =======================
HARD_BACKENDS = ["glot500", "labse", "laser", "sonar", "xlmr"]  # e.g. ["llama31_noprompt", "llama31_noprompt"] or None for all
# =======================


def cluster_based(
    representations: np.ndarray,
    n_cluster: int = 7,
    n_pc: int = 12,
    hidden_size: int = 768,
    seed: int = 42,
) -> np.ndarray:
    # kmeans init as in your old script
    centroids, labels = clst.vq.kmeans2(
        representations,
        n_cluster,
        minit="points",
        missing="warn",
        check_finite=True,
        seed=seed,
    )

    # ---- robust cluster mean computation (no div-by-zero) ----
    sums = np.zeros((n_cluster, hidden_size), dtype=np.float32)
    counts = np.zeros((n_cluster,), dtype=np.int32)

    for i, lab in enumerate(labels):
        sums[lab] += representations[i]
        counts[lab] += 1

    cluster_means = np.zeros_like(sums)
    for c in range(n_cluster):
        if counts[c] > 0:
            cluster_means[c] = sums[c] / float(counts[c])

    # subtract cluster mean
    zero_mean = representations - cluster_means[labels]

    # allocate output
    post_rep = np.zeros_like(representations)

    # PCA + removal per cluster
    for c in range(n_cluster):
        idxs = np.nonzero(labels == c)[0]
        if len(idxs) == 0:
            continue

        cluster_vecs = zero_mean[idxs]
        pca = PCA()
        pca.fit(cluster_vecs)
        comps = pca.components_

        for idx in idxs:
            v = zero_mean[idx]
            sub = np.zeros_like(v)
            for j in range(min(n_pc, comps.shape[0])):
                sub += np.dot(v, comps[j]) * comps[j]
            post_rep[idx] = v - sub

    return post_rep


def read_vec(path: Path):
    """Read PaSeMiLL-style .vec file."""
    with path.open("r", encoding="utf-8") as f:
        header = f.readline().strip()
        if not header:
            raise ValueError(f"empty header in {path}")
        n_str, dim_str = header.split()
        n, dim = int(n_str), int(dim_str)

        ids = []
        vecs = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            ids.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])

    arr = np.array(vecs, dtype=np.float32)
    if arr.shape != (n, dim):
        raise ValueError(
            f"shape mismatch in {path}: got {arr.shape}, header says {(n, dim)}"
        )
    return ids, arr


def write_vec(path: Path, ids, vecs: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(ids)} {vecs.shape[1]}\n")
        for sid, v in zip(ids, vecs):
            f.write(sid + " " + " ".join(f"{x:.6f}" for x in v.tolist()) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pivot",
        default="tur_Latn",
        help="pivot lang folder under results/, e.g. deu_Latn",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="force recompute + overwrite CBIE files",
    )
    ap.add_argument(
        "--recompute-if-stale",
        action="store_true",
        help="recompute only if RAW input is newer than CBIE output",
    )
    args = ap.parse_args()

    # assume script is in .../code, project root is parent
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]  # .../PaSeMiLL
    results_root = project_root / "results"

    raw_root = results_root / args.pivot / "raw"
    cbie_root = results_root / args.pivot / "cbie"

    if not raw_root.exists():
        raise SystemExit(f"[ERR] raw root not found: {raw_root}")

    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] pivot        = {args.pivot}")
    print(f"[INFO] scanning models under {raw_root}")
    print(f"[INFO] overwrite    = {args.overwrite}")
    print(f"[INFO] stale-recomp  = {args.recompute_if_stale}")
    print(f"[INFO] hard_backends = {HARD_BACKENDS}")

    # choose which backends to process
    if HARD_BACKENDS is None:
        backend_dirs = [p for p in raw_root.iterdir()]
    else:
        backend_dirs = [raw_root / name for name in HARD_BACKENDS]

    # loop over selected backends
    for backend_dir in backend_dirs:
        if not backend_dir.exists():
            print(f"[WARN] backend not found: {backend_dir} (skipping)")
            continue
        if not backend_dir.is_dir():
            continue

        backend_name = backend_dir.name  # e.g. glot500, labse, xlmr, llama31_noprompt
        raw_emb_root = backend_dir / "embeddings" / "doc"
        if not raw_emb_root.exists():
            print(f"[WARN] {backend_name}: no embeddings/doc/, skipping")
            continue

        cbie_emb_root = cbie_root / backend_name / "embeddings" / "doc"
        print(f"\n[INFO] processing backend: {backend_name}")
        print(f"[INFO] raw_emb_root  = {raw_emb_root}")
        print(f"[INFO] cbie_emb_root = {cbie_emb_root}")

        for vec_path in raw_emb_root.rglob("*.vec"):
            rel = vec_path.relative_to(raw_emb_root)
            out_path = cbie_emb_root / rel

            # ---------- SMART SKIP ----------
            if out_path.is_file() and out_path.stat().st_size > 0 and not args.overwrite:
                if args.recompute_if_stale:
                    # recompute only if raw is newer than cbie
                    if vec_path.stat().st_mtime <= out_path.stat().st_mtime:
                        print(f"[skip cbie] up-to-date: {out_path}")
                        continue
                else:
                    # only-missing mode
                    print(f"[skip cbie] exists: {out_path}")
                    continue

            # recompute
            print(f"[CBIE] {backend_name}: {vec_path} -> {out_path}")

            ids, vecs = read_vec(vec_path)

            cbie_vecs = cluster_based(
                vecs,
                n_cluster=7,
                n_pc=12,
                hidden_size=vecs.shape[1],
            )

            write_vec(out_path, ids, cbie_vecs)

    print("\n[DONE] CBIE export finished.")


if __name__ == "__main__":
    main()
