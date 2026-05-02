#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

# ---- Projekt-Root sauber bestimmen ----
HERE = Path(__file__).resolve()          # .../src/run_all_models.py
ROOT = HERE.parent.parent                # .../ (outliers)
EMBS_ROOT = ROOT / "embs"
SRC_DIR = ROOT / "src"

def run(cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True)

# adjust these to your real folders (jetzt auf Basis von EMBS_ROOT)
MODELS = {
    "glot500": {
        "raw":      EMBS_ROOT / "bucc" / "cis-lmu" / "glot500-base" / "7",
        # "cbie":     EMBS_ROOT / "bucc" / "cis-lmu" / "glot500-base" / "cbie",
        # "whitened": EMBS_ROOT / "bucc" / "cis-lmu" / "glot500-base" / "whitened",
    },
    "labse": {
        "raw":      EMBS_ROOT / "bucc" / "sentence-transformers" / "LaBSE" / "7",
        # "cbie":     EMBS_ROOT / "bucc" / "sentence-transformers" / "LaBSE" / "cbie",
        # "whitened": EMBS_ROOT / "bucc" / "sentence-transformers" / "LaBSE" / "whitened",
    },
    "sonar": {
        "raw":      EMBS_ROOT / "bucc" / "sonar" / "7",
        # "cbie":     EMBS_ROOT / "bucc" / "xlm-roberta-base" / "cbie",
        # "whitened": EMBS_ROOT / "bucc" / "xlm-roberta-base" / "whitened",
    },
    #"qwen3": {
     #   "raw":      EMBS_ROOT / "bucc" / "qwen3" / "7",
    #},
    #"llama31": {
     #   "raw":      EMBS_ROOT / "bucc" / "llama31" / "7",
    #},
}

def main():
    print(f"[INFO] ROOT={ROOT}")
    print(f"[INFO] EMBS_ROOT={EMBS_ROOT}")

    for model, variants in MODELS.items():
        for variant_name, emb_dir in variants.items():
            emb_dir = Path(emb_dir)
            if not emb_dir.is_dir():
                print(f"[warn] skipping {model}/{variant_name}, dir not found: {emb_dir}")
                continue

            print("\n==============================")
            print(f"🚀 Processing {model} ({variant_name})")
            print("==============================")

            # 1) create centroids
            run([
                sys.executable,  # benutzt das aktuelle Python (gut für Conda)
                str(SRC_DIR / "create_centroids.py"),
                "--emb_dir", str(emb_dir),
            ])

            # 2) compute distances/plots
            centroids_npz = SRC_DIR / "centroids" / variant_name / model / "centroids.npz"
            if not centroids_npz.is_file():
                print(f"❌ Missing centroids file: {centroids_npz}", file=sys.stderr)
                continue

            run([
                sys.executable,
                str(SRC_DIR / "compute_lang_centroid_distances_and_plot.py"),
                "--centroids_npz", str(centroids_npz),
            ])

    print("\n✅ All models/variants processed!")

if __name__ == "__main__":
    main()
