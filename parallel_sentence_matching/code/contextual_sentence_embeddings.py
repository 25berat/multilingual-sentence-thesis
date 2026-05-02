#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from tqdm import tqdm
import torch

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM  # <-- mmBERT

import utils as utils

# SONAR import (kommt aus sonar-space)
try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    HAS_SONAR = True
except ImportError:
    HAS_SONAR = False

# LASER import
try:
    from laser_encoders import LaserEncoder
    HAS_LASER = True
except ImportError:
    HAS_LASER = False


### MODIFY PATH ###
PRETRAINING_PATH = "/dss/dsshome1/06/ge65hon2/projects/PaSeMiLL/code/model/mmbert_ft_finalonly"
QWEN3_MODEL_ID = "Qwen/Qwen3-Embedding-8B"
LLAMA31_MODEL_ID = "/dss/dsshome1/06/ge65hon2/models/llama-3.1-8b"
STUDENTTEACHER_PATH = "/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge65hon2/student_teacher_run_debug/run_2026-04-16_18-18-57/model/final"


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=True,
        help="Sentence file (format: <ID>\\t<sentence>)",
    )
    parser.add_argument(
        "-o", "--output_file", type=str, required=True, help="Output file path"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        choices=[
            "xlmr",
            "glot500",
            "pretrained",
            "studentteacher",
            "mmbert",
            "labse",
            "sonar",
            "laser",
            "qwen3",
            "llama31_noprompt",
        ],
        help="Embedding model",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--truncate_dim", type=int, default=768, help="0 = keep full dim")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None))

    # Layer selection (optional)
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Hidden-state layer index to use. If omitted: xlmr/glot500/pretrained=8, mmbert=21.",
    )

    # optional: normalize sentence embeddings
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize sentence vectors before writing",
    )

    return parser.parse_args()


# -------------------------------
# I/O helpers
# -------------------------------
def write_header_and_flush(path, n, dim):
    with open(path, "w", encoding="utf8") as out_text:
        out_text.write(f"{n} {dim}\n")


# -------------------------------
# Pooling
# -------------------------------
def masked_mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden: (B, T, D)
    attention_mask: (B, T) with 1 for real tokens, 0 for padding
    Returns: (B, D)
    """
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)  # (B,T,1)
    summed = (last_hidden * mask).sum(dim=1)                         # (B,D)
    denom = mask.sum(dim=1).clamp(min=1e-6)                          # (B,1)
    return summed / denom


def masked_mean_pool_drop_ends(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Drops first + last active token, then does masked mean.
    """
    B, T, _ = hidden.shape
    attn = attention_mask.clone()
    lengths = attn.sum(dim=1)

    # drop first position
    if T > 0:
        attn[:, 0] = 0

    # drop last active token per row
    valid = lengths >= 3
    if valid.any():
        last_idx = (lengths - 1).clamp(min=0)
        rows = torch.arange(B, device=hidden.device)
        attn[rows[valid], last_idx[valid]] = 0

    return masked_mean_pool(hidden, attn)


# -------------------------------
# HF encoder loader (XLM-R / Glot500 / Pretrained)
# -------------------------------
class HFEmbeddingLoader:
    def __init__(self, model_id: str, device: torch.device, layer: int, max_length: int, hf_token=None):
        self.model_id = model_id
        self.device = device
        self.layer = layer
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModel.from_pretrained(model_id, token=hf_token, output_hidden_states=True)

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        out = self.model(**inputs)
        hidden_states = out["hidden_states"]

        if self.layer >= len(hidden_states):
            raise ValueError(f"Requested layer={self.layer}, but only {len(hidden_states)} layers returned")

        h = hidden_states[self.layer].float()  # force fp32 for stable pooling
        attn = inputs["attention_mask"]
        return h, attn


# -------------------------------
# mmBERT loader (stable + memory-friendly)
# -------------------------------
class MMBertEmbeddingLoader:
    def __init__(
        self,
        device: torch.device,
        layer: int,
        max_length: int,
        hf_token: str | None = None,
    ):
        self.model_id = "jhu-clsp/mmBERT-base"
        self.device = device
        self.layer = layer
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=hf_token,
            trust_remote_code=True,
        )

        # Load MLM wrapper, but we will run only the base encoder (avoids logits OOM)
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_id,
            token=hf_token,
            trust_remote_code=True,
            use_safetensors=True,
            output_hidden_states=True,
        )

        self.model.eval()
        self.model.to(self.device)

        # Try to locate the base encoder reliably
        self.base = getattr(self.model, "base_model", None)
        if self.base is None:
            self.base = getattr(self.model, "model", None)
        if self.base is None:
            raise RuntimeError("Could not find base encoder on the MLM model wrapper (expected .base_model or .model).")

        self.base.eval()
        self.base.to(self.device)

    @torch.no_grad()
    def encode_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Force full precision for stability (prevents weird huge values)
        # Also explicitly request hidden states.
        out = self.base(**inputs, output_hidden_states=True, return_dict=True)

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("mmBERT base model did not return hidden_states (need output_hidden_states=True).")

        if self.layer >= len(hidden_states):
            raise ValueError(
                f"Requested layer={self.layer}, but model returned only {len(hidden_states)} hidden_states."
            )

        h = hidden_states[self.layer].float()  # <-- critical: stabilize pooling
        attn = inputs["attention_mask"]
        return h, attn


# -------------------------------
# Common writer for HF-style encoders
# -------------------------------
def to_sentence_embeddings_hfstyle(
    path: str,
    sentence_list: list[str],
    loader,
    batch_size: int,
    truncate_dim: int,
    normalize: bool,
):
    # infer dim
    h_test, _ = loader.encode_batch(["hello"])
    full_dim = int(h_test.shape[-1])
    dim = full_dim if (truncate_dim is None or truncate_dim == 0) else int(truncate_dim)

    # split ids/texts
    ids, texts = [], []
    for line in sentence_list:
        line = line.rstrip("\n")
        parts = line.split("\t", 1)
        ids.append(parts[0])
        texts.append(parts[1] if len(parts) == 2 else "")

    n = len(texts)
    write_header_and_flush(path, n, dim)

    with open(path, "a", encoding="utf8") as out_f:
        for start in tqdm(range(0, n, batch_size), desc=f"HF embed ({getattr(loader,'model_id','hf')})"):
            end = min(start + batch_size, n)
            batch_ids = ids[start:end]
            batch_txt = texts[start:end]

            h, attn = loader.encode_batch(batch_txt)

            # drop ends then mean-pool
            sent_emb = masked_mean_pool_drop_ends(h, attn)

            if dim != full_dim:
                sent_emb = sent_emb[:, :dim]

            if normalize:
                sent_emb = torch.nn.functional.normalize(sent_emb, p=2, dim=1)

            sent_emb = sent_emb.to(torch.float32).cpu().numpy()

            for sid, vec in zip(batch_ids, sent_emb):
                out_f.write(sid + " " + " ".join(f"{x:.6f}" for x in vec.tolist()) + "\n")


# -------------------------------
# Model dispatch
# -------------------------------
def to_xlmr_sentence_embeddings(path, sentence_list, model_name, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "xlmr":
        hf_id = "xlm-roberta-base"
        default_layer = 8
        loader = HFEmbeddingLoader(hf_id, device, args.layer or default_layer, args.max_length, args.hf_token)

    elif model_name == "glot500":
        hf_id = "cis-lmu/glot500-base"
        default_layer = 8
        loader = HFEmbeddingLoader(hf_id, device, args.layer or default_layer, args.max_length, args.hf_token)

    elif model_name == "pretrained":
        hf_id = PRETRAINING_PATH
        default_layer = 8
        loader = HFEmbeddingLoader(hf_id, device, args.layer or default_layer, args.max_length, args.hf_token)

    elif model_name == "mmbert":
        default_layer = 21
        layer = args.layer if args.layer is not None else default_layer
        loader = MMBertEmbeddingLoader(device=device, layer=layer, max_length=args.max_length, hf_token=args.hf_token)
        
    elif model_name == "studentteacher":
        hf_id = STUDENTTEACHER_PATH
        default_layer = 8
        loader = HFEmbeddingLoader(hf_id, device, args.layer or default_layer, args.max_length, args.hf_token)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    to_sentence_embeddings_hfstyle(
        path=path,
        sentence_list=sentence_list,
        loader=loader,
        batch_size=args.batch_size,
        truncate_dim=args.truncate_dim,
        normalize=args.normalize,
    )


# -------------------------------
# Main
# -------------------------------
def main():
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        input_file = f.read()
    split_file = utils.text_to_line(input_file)

    model_name = args.model_name
    print(f"Model to use: {model_name}")

    if model_name in ["xlmr","studentteacher", "glot500", "pretrained", "mmbert"]:
        to_xlmr_sentence_embeddings(args.output_file, split_file, model_name, args)
    else:
        raise ValueError(f"Model {model_name} not included in this old revert version.")

    print(f"Done. Saved embeddings to {args.output_file}")


if __name__ == "__main__":
    main()
