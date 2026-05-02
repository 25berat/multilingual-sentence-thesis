#!/usr/bin/env python3
import argparse, os, operator
from collections import defaultdict

import numpy as np
import torch
from torch import nn

# ---------- helpers ----------
def detect_model_from_path(p: str) -> str:
    s = p.lower()
    if "glot500" in s: return "glot500"
    if "labse"   in s: return "labse"
    if "xlm"     in s: return "sonar"
    return "other"

def load_emb(path: str) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in ("emb", "embs", "embeddings"):
            if k in obj:
                obj = obj[k]; break
    x = obj.numpy() if hasattr(obj, "numpy") else np.asarray(obj)
    if x.ndim == 1: x = x.reshape(1, -1)
    return torch.from_numpy(x).float()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Dimension sensitivity on parallel (TGT, ENG) sentence pairs.")
    ap.add_argument("--emb_dir", required=True,
                    help="Root: ./embs/bucc/<model-path>/<layer> (contains <lang>/emb.pt folders)")
    ap.add_argument("--eng_lang", default="eng_Latn",
                    help="Folder name for English (default: eng_Latn)")
    ap.add_argument("--hidden_size", type=int, default=768,
                    help="Embedding dim (used for loop limit; inferred if not given)")
    ap.add_argument("--language", default="", help="If set, only that language vs English")
    ap.add_argument("--layer_name", default="", help="Optional label for output filenames (e.g., 7)")
    args = ap.parse_args()

    model = detect_model_from_path(args.emb_dir)
    out_dir = os.path.join("src", "cosine_sensitivity", model)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"layer{args.layer_name}" if args.layer_name else "layer"
    out_file = os.path.join(out_dir, f"dim_sensitivity_{tag}.tsv")

    # collect languages (subfolders that have emb.pt), exclude English
    langs = []
    for name in os.listdir(args.emb_dir):
        lang_dir = os.path.join(args.emb_dir, name)
        if not os.path.isdir(lang_dir): continue
        if name == args.eng_lang: continue
        if os.path.isfile(os.path.join(lang_dir, "emb.pt")):
            langs.append(name)
    langs = sorted(langs)

    if args.language:
        if args.language not in langs:
            raise FileNotFoundError(f"{args.language} not found under {args.emb_dir}")
        langs = [args.language]

    # load English once
    eng_path = os.path.join(args.emb_dir, args.eng_lang, "emb.pt")
    if not os.path.isfile(eng_path):
        raise FileNotFoundError(f"English file not found: {eng_path}")
    eng = load_emb(eng_path)

    cos = nn.CosineSimilarity(dim=1)

    def mean_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # assume rows are aligned parallel pairs
        n = min(a.shape[0], b.shape[0])
        a = a[:n]; b = b[:n]
        return cos(a, b).mean()

    differences = defaultdict(float)
    n_langs = 0

    for lang in langs:
        print(f"\n🌍 Language: {lang}")
        tgt_path = os.path.join(args.emb_dir, lang, "emb.pt")
        if not os.path.isfile(tgt_path):
            print(f"  - skip (missing): {tgt_path}")
            continue

        tgt = load_emb(tgt_path)

        # infer dim if needed
        hidden = args.hidden_size or tgt.shape[1]
        if tgt.shape[1] != eng.shape[1]:
            raise ValueError(f"Dim mismatch: {lang} has {tgt.shape[1]} vs ENG {eng.shape[1]}")

        with torch.no_grad():
            orig = mean_cos(tgt, eng).item()
            print(f"  Original cosine similarity: {orig:.6f}")

            # transpose once to zero dims efficiently
            tgt_T = tgt.T.clone()
            eng_T = eng.T.clone()
            for d in range(hidden):
                val_t, val_e = tgt_T[d].clone(), eng_T[d].clone()
                tgt_T[d].zero_(); eng_T[d].zero_()
                new_tgt, new_eng = tgt_T.T, eng_T.T
                removed = mean_cos(new_tgt, new_eng).item()
                differences[d] += (removed - orig)
                # restore for next dim
                tgt_T[d] = val_t; eng_T[d] = val_e

        n_langs += 1

    if n_langs == 0:
        raise RuntimeError("No languages processed.")

    # average and sort
    for k in differences:
        differences[k] /= n_langs
    sorted_d = dict(sorted(differences.items(), key=operator.itemgetter(1), reverse=True))

    # save
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("dim\tdelta_cosine\n")
        for k, v in sorted_d.items():
            f.write(f"{k}\t{v:.6f}\n")

    print(f"\n[✓] Model: {model}")
    print(f"[✓] Languages processed: {n_langs}")
    print(f"[✓] Saved dimension sensitivity to: {out_file}")
    print(f"[i] Top 5 dims: {list(sorted_d.items())[:5]}")

if __name__ == "__main__":
    main()
