#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import argparse
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

ENCODING = "utf-8"


# ============================================================
# Plugin registry (SIMPLE + HARD TO BREAK)
# plugin_name -> (backend_type, identifier)
# backend_type: "hf" or "laser"
# ============================================================
PLUGINS = {
    "glot500": ("hf", "cis-lmu/glot500-base"),
    "xlmr_base": ("hf", "xlm-roberta-base"),
    "labse": ("hf", "sentence-transformers/LaBSE"),

    # SONAR only if you have a HF repo id WITH a tokenizer.
    # If your sonar checkpoint differs, pass the HF id directly via --model <repo_id>.
     #"sonar": ("hf", "facebook/sonar-..."),

    # Classic LASER (non-HF)
    #"laser": ("laser", "laser"),
}


def sanitize_model_tag(tag: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", tag)


# ============================================================
# HF backend
# ============================================================
def load_hf_tokenizer(model_id: str):
    print(f"[INFO] Loading HF tokenizer: {model_id}")
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    except Exception as e_fast:
        print(f"[WARN] use_fast=True failed ({e_fast}). Retrying with use_fast=False...")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    # We are not running a model; we want full sequences.
    tok.model_max_length = 10**30
    return tok


def hf_tokenize_text_to_ids(tokenizer, text: str) -> List[int]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # No truncation, no special tokens
    return tokenizer.encode(text, add_special_tokens=False)


def compute_stats_for_file_hf(tokenizer, file_path: Path, stream_by_lines: bool) -> Tuple[int, int, float]:
    if stream_by_lines:
        token_ids: List[int] = []
        with file_path.open("r", encoding=ENCODING, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                token_ids.extend(hf_tokenize_text_to_ids(tokenizer, line))
    else:
        text = file_path.read_text(encoding=ENCODING, errors="replace").strip()
        if not text:
            return 0, 0, 0.0
        token_ids = hf_tokenize_text_to_ids(tokenizer, text)

    tokens = len(token_ids)
    if tokens == 0:
        return 0, 0, 0.0
    types = len(set(token_ids))
    return tokens, types, types / tokens


# ============================================================
# Classic LASER backend (Moses + BPE)
# ============================================================
class LaserTokenizer:
    """
    Classic LASER tokenization:
      - Moses punct normalization + tokenization
      - BPE via subword-nmt apply_bpe
    Requires:
      pip install sacremoses subword-nmt
    """
    def __init__(self, bpe_codes_path: Path, moses_lang: str = "en"):
        try:
            from sacremoses import MosesPunctNormalizer, MosesTokenizer
        except Exception as e:
            raise RuntimeError(
                "LASER backend requires 'sacremoses'. Install: pip install sacremoses\n"
                f"Original error: {e}"
            )
        try:
            from subword_nmt.apply_bpe import BPE
        except Exception as e:
            raise RuntimeError(
                "LASER backend requires 'subword-nmt'. Install: pip install subword-nmt\n"
                f"Original error: {e}"
            )

        if not bpe_codes_path.exists():
            raise FileNotFoundError(
                f"LASER BPE codes not found: {bpe_codes_path}\n"
                "Provide via --laser_codes <path_to_bpe.codes>"
            )

        self.mpn = MosesPunctNormalizer()
        self.mt = MosesTokenizer(lang=moses_lang)
        with bpe_codes_path.open("r", encoding="utf-8") as f:
            self.bpe = BPE(f)

    def tokenize_to_pieces(self, text: str) -> List[str]:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self.mpn.normalize(text)
        toks = self.mt.tokenize(text, return_str=True)
        bpe_line = self.bpe.process_line(toks).strip()
        return bpe_line.split() if bpe_line else []


def compute_stats_for_file_laser(laser_tok: LaserTokenizer, file_path: Path, stream_by_lines: bool) -> Tuple[int, int, float]:
    if stream_by_lines:
        all_tokens = 0
        all_types = set()
        with file_path.open("r", encoding=ENCODING, errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pieces = laser_tok.tokenize_to_pieces(line)
                all_tokens += len(pieces)
                all_types.update(pieces)
        tokens = all_tokens
        types = len(all_types)
    else:
        text = file_path.read_text(encoding=ENCODING, errors="replace").strip()
        if not text:
            return 0, 0, 0.0
        pieces = laser_tok.tokenize_to_pieces(text)
        tokens = len(pieces)
        types = len(set(pieces))

    if tokens == 0:
        return 0, 0, 0.0
    return tokens, types, types / tokens


# ============================================================
# Output: CSV + PNG table
# ============================================================
def save_sorted_table_png(df: pd.DataFrame, png_path: Path, top_n: Optional[int], sort_by: str, descending: bool):
    df2 = df.sort_values(sort_by, ascending=not descending).reset_index(drop=True)
    if top_n is not None and top_n > 0:
        df2 = df2.head(top_n)

    df_show = df2.copy()
    df_show["tokens"] = df_show["tokens"].map(lambda x: f"{int(x):,}")
    df_show["types"] = df_show["types"].map(lambda x: f"{int(x):,}")
    df_show["ttr"] = df_show["ttr"].map(lambda x: f"{x:.6f}")

    nrows = len(df_show)
    fig_w = 10
    fig_h = max(2.5, 0.35 * (nrows + 1))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")

    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Model resolver
# ============================================================
def resolve_model(model_arg: str) -> Tuple[str, str, str]:
    """
    Returns (backend_type, identifier, out_tag)

    - If model_arg is a plugin key, map it.
    - Otherwise assume it's a HF model id.
    """
    if model_arg in PLUGINS:
        backend, ident = PLUGINS[model_arg]
        out_tag = model_arg if backend == "laser" else sanitize_model_tag(ident)
        return backend, ident, out_tag

    # direct HF model id
    backend, ident = "hf", model_arg
    out_tag = sanitize_model_tag(model_arg)
    return backend, ident, out_tag


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Compute (subword) TTR for each language file using selected tokenizer.")
    parser.add_argument("--input_dir", type=str,
                        default=r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\unshuffled_custom_data")
    parser.add_argument("--out_root", type=str,
                        default=r"C:\Users\berat\PycharmProjects\PaSeMiLL\results\TTR")
    parser.add_argument("--model", type=str, default="glot500",
                        help="Plugin key (glot500/xlmr_base/xlmr_large/labse/laser) or HF model id.")
    parser.add_argument("--stream_by_lines", action="store_true",
                        help="Tokenize line-by-line (memory friendly).")

    parser.add_argument("--sort_by", type=str, default="ttr",
                        choices=["language", "tokens", "types", "ttr"])
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--top_n", type=int, default=60)

    # LASER options
    parser.add_argument("--laser_codes", type=str, default="",
                        help="Path to classic LASER BPE codes (required if --model laser).")
    parser.add_argument("--laser_moses_lang", type=str, default="en",
                        help="Moses language code for LASER tokenization (default en).")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    files = sorted(input_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {input_dir}")

    backend, identifier, out_tag = resolve_model(args.model)

    out_root = Path(args.out_root)
    out_dir = out_root / out_tag
    out_csv = out_dir / "ttr_all_languages.csv"
    out_png = out_dir / f"table_sorted_by_{args.sort_by}.png"

    print(f"[INFO] Requested model: {args.model}")
    print(f"[INFO] Backend type: {backend}")
    print(f"[INFO] Resolved identifier: {identifier}")
    print(f"[INFO] Output dir: {out_dir}")

    # Load backend
    hf_tok = None
    laser_tok = None
    if backend == "hf":
        hf_tok = load_hf_tokenizer(identifier)
    elif backend == "laser":
        if not args.laser_codes:
            raise ValueError(
                "You selected --model laser but did not provide --laser_codes.\n"
                "Example:\n"
                "  python tokenizer_TTR.py --model laser --laser_codes C:\\path\\to\\bpe.codes\n"
            )
        laser_tok = LaserTokenizer(Path(args.laser_codes), moses_lang=args.laser_moses_lang)
    else:
        raise RuntimeError(f"Unknown backend: {backend}")

    rows = []
    for fp in files:
        lang = fp.stem

        if backend == "hf":
            tokens, types, ttr = compute_stats_for_file_hf(hf_tok, fp, stream_by_lines=args.stream_by_lines)
        else:
            tokens, types, ttr = compute_stats_for_file_laser(laser_tok, fp, stream_by_lines=args.stream_by_lines)

        rows.append({"language": lang, "tokens": tokens, "types": types, "ttr": ttr})

    df = pd.DataFrame(rows).sort_values("language").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")

    top_n = None if args.top_n <= 0 else args.top_n
    save_sorted_table_png(df, out_png, top_n=top_n, sort_by=args.sort_by, descending=not args.ascending)

    print(f"[OK] Processed languages: {len(df)}")
    print(f"[OK] Saved CSV: {out_csv}")
    print(f"[OK] Saved PNG: {out_png}")
    print(df.sort_values(args.sort_by, ascending=args.ascending).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
