#!/usr/bin/env python3
"""
Auto-whitening for PaSeMiLL .vec embeddings (HARDCODED to llama31_noprompt).

- Input:  <project_root>/results/deu_Latn/raw/<backend>/embeddings/doc/**/*.vec
- Output: <project_root>/results/deu_Latn/whitened/<backend>/embeddings/doc/**/*.vec
- ALWAYS recomputes whitening for all files (overwrites existing outputs).

Whitening params: epsilon = 1e-5.

CHANGE THIS WHEN NEEDED:
    HARD_BACKENDS = ["llama31_noprompt"]              # only llama31_noprompt
    HARD_BACKENDS = ["llama31_noprompt","llama31_noprompt"]
    HARD_BACKENDS = None                    # all backends
"""

from pathlib import Path
import numpy as np

# =======================
# CHANGE THIS WHEN NEEDED
# =======================
HARD_BACKENDS = ["glot500", "labse", "laser", "sonar", "xlmr"]  # or None for all
# =======================


# ------------------------------------------------------------
# whitening adapted to (n_samples, dim) data
# ------------------------------------------------------------
def whitening(representations: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    ZCA(-like) whitening:
      1) center
      2) covariance over features
      3) eigendecomp
      4) apply W
    """
    mu = representations.mean(axis=0, keepdims=True)  # (1, d)
    X = representations - mu                           # (n, d)

    cov = np.cov(X, rowvar=False)                      # (d, d)

    vals, vecs = np.linalg.eigh(cov)
    vals = np.real(vals)
    vecs = np.real(vecs)

    W = vecs @ np.diag(1.0 / np.sqrt(vals + epsilon)) @ vecs.T
    Xw = X @ W.T
    return Xw.astype(np.float32)


# ------------------------------------------------------------
# PaSeMiLL .vec I/O
# ------------------------------------------------------------
def read_vec(path: Path):
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
        raise ValueError(f"shape mismatch in {path}: got {arr.shape}, header says {(n, dim)}")
    return ids, arr


def write_vec(path: Path, ids, vecs: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{len(ids)} {vecs.shape[1]}\n")
        for sid, v in zip(ids, vecs):
            f.write(sid + " " + " ".join(f"{x:.6f}" for x in v.tolist()) + "\n")


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    results_root = project_root / "results" / "tur_Latn"

    raw_root = results_root / "raw"
    whiten_root = results_root / "whitened"

    if not raw_root.exists():
        raise SystemExit(f"[ERR] raw root not found: {raw_root}")

    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] results_root = {results_root}")
    print(f"[INFO] raw_root     = {raw_root}")
    print(f"[INFO] whiten_root  = {whiten_root}")
    print(f"[INFO] hard_backends= {HARD_BACKENDS}")

    # choose backends
    if HARD_BACKENDS is None:
        backend_dirs = [p for p in raw_root.iterdir() if p.is_dir()]
    else:
        backend_dirs = [raw_root / name for name in HARD_BACKENDS]

    for backend_dir in backend_dirs:
        if not backend_dir.exists():
            print(f"[WARN] backend not found: {backend_dir} (skipping)")
            continue
        if not backend_dir.is_dir():
            continue

        backend_name = backend_dir.name
        raw_emb_root = backend_dir / "embeddings" / "doc"
        if not raw_emb_root.exists():
            print(f"[WARN] {backend_name}: no embeddings/doc/, skipping")
            continue

        out_emb_root = whiten_root / backend_name / "embeddings" / "doc"
        print(f"\n[INFO] processing backend: {backend_name}")
        print(f"[INFO] raw_emb_root    = {raw_emb_root}")
        print(f"[INFO] whiten_emb_root = {out_emb_root}")

        for vec_path in raw_emb_root.rglob("*.vec"):
            rel = vec_path.relative_to(raw_emb_root)
            out_path = out_emb_root / rel

            print(f"[WHITEN] {backend_name}: {vec_path} -> {out_path} (overwrite)")

            ids, vecs = read_vec(vec_path)
            vecs_w = whitening(vecs, epsilon=1e-5)
            write_vec(out_path, ids, vecs_w)

    print("\n[DONE] Whitening export finished.")


if __name__ == "__main__":
    main()
