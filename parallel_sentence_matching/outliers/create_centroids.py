#!/usr/bin/env python3
import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch


def detect_model_from_path(p: str) -> str:
    p = p.lower()
    if "llama31" in p:
        return "llama31"
    if "qwen3" in p:
        return "qwen3"
    if "xlm" in p:      # xlm-roberta-base
        return "sonar"
    if "laser" in p:
        return "laser"
    if "sonar" in p:
        return "sonar"
    if "glot500" in p:
        return "glot500"
    if "labse" in p:
        return "labse"

    return "other"


def detect_variant_from_path(p: str) -> str:
    p_low = p.lower()
    if "cbie" in p_low:
        return "cbie"
    if "whitened" in p_low or "white" in p_low:
        return "whitened"
    return "raw"


def process_one_emb_dir(emb_dir: Path, emb_name: str = "emb.pt", recursive: bool = False) -> None:
    emb_dir = emb_dir.resolve()
    if not emb_dir.is_dir():
        print(f"[WARN] emb_dir does not exist, skipping: {emb_dir}")
        return

    model_name = detect_model_from_path(str(emb_dir))
    variant = detect_variant_from_path(str(emb_dir))

    # --- FIX: anchor centroids dir on script location, not CWD ---
    here = Path(__file__).resolve()        # .../outliers/src/create_centroids.py
    src_dir = here.parent                  # .../outliers/src
    centroids_root = src_dir / "centroids" # .../outliers/src/centroids

    out_dir = centroids_root / variant / model_name
    out_file = out_dir / "centroids.npz"
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(emb_dir / ("**" if recursive else "*") / emb_name)
    files = sorted(glob(pattern, recursive=recursive))
    if not files:
        print(f"[WARN] No embedding files found with pattern: {pattern}")
        return

    centroids = {}
    for f in files:
        f_path = Path(f)
        lang = f_path.parent.name  # language folder
        obj = torch.load(f_path, map_location="cpu")
        if isinstance(obj, dict):
            for key in ("emb", "embs", "embeddings"):
                if key in obj:
                    obj = obj[key]
                    break
        X = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        centroids[lang] = X.mean(axis=0)

    np.savez_compressed(out_file, **centroids)
    print(f"[✓] emb_dir: {emb_dir}")
    print(f"[✓] Detected model:   {model_name}")
    print(f"[✓] Detected variant: {variant}")
    print(f"[✓] Found {len(files)} language files")
    print(f"[✓] Saved {len(centroids)} centroids to {out_file}")
    print()


def get_default_emb_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    root = here.parent.parent              # .../outliers
    emb_root = root / "embs" / "bucc"

    paths = [
        #emb_root / "cis-lmu" / "glot500-base" / "7",
        #emb_root / "sentence-transformers" / "LaBSE" / "7",
        #emb_root / "xlm-roberta-base" / "7",
        #emb_root / "laser" / "7",
#emb_root / "sonar" / "7",
        emb_root / "qwen3" / "7",
        emb_root / "llama31" / "7",
    ]
    return paths


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Compute language centroids.\n"
            "- If --emb_dir is given: process exactly that directory.\n"
            "- If no --emb_dir: process all default model dirs (glot500, LaBSE, XLM-R, LASER, SONAR)."
        )
    )
    ap.add_argument(
        "--emb_dir",
        help="Folder containing per-language subfolders with emb.pt files "
             "(if omitted, all default models are processed)",
    )
    ap.add_argument("--emb_name", default="emb.pt")
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    if args.emb_dir:
        process_one_emb_dir(Path(args.emb_dir), emb_name=args.emb_name, recursive=args.recursive)
    else:
        print("[INFO] No --emb_dir given → AUTO-MODE over default models.\n")
        for emb_dir in get_default_emb_dirs():
            process_one_emb_dir(emb_dir, emb_name=args.emb_name, recursive=args.recursive)
        print("[DONE] All default models processed.")


if __name__ == "__main__":
    main()
