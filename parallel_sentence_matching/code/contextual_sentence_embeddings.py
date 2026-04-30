#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import torch
from tqdm import tqdm
from transformers import (
    XLMRobertaModel,
    XLMRobertaTokenizer,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from sentence_transformers import SentenceTransformer
import utils as utils

# SONAR import (kommt aus sonar-space)
try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    HAS_SONAR = True
except ImportError:
    HAS_SONAR = False

# LASER import
try:
    # this is the package you said you were installing
    from laser_encoders import LaserEncoder
    HAS_LASER = True
except ImportError:
    HAS_LASER = False

### MODIFY PATH ###
PRETRAINING_PATH = r"C:\Users\berat\PycharmProjects\PaSeMiLL\code\model\mmbert_ft"


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
        # 👉 hier LASER ergänzen
        choices=["xlmr", "glot500", "pretrained_partial", "labse", "sonar", "laser", "qwen3", "llama31_noprompt", "mmbert"],
        help="Embedding model",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--truncate_dim", type=int, default=768, help="0 = keep full dim")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", None))
    return parser.parse_args()


# -------------------------------
# Embedding manager (XLM-R etc.)
# -------------------------------
class EmbeddingLoader(object):
    def __init__(self, model: str, device=torch.device("cpu"), layer: int = 8):
        TR_Models = {"xlm-roberta-base": (XLMRobertaModel, XLMRobertaTokenizer)}

        self.model = model
        self.device = device
        self.layer = layer
        self.emb_model = None
        self.tokenizer = None

        if model in TR_Models:
            model_class, tokenizer_class = TR_Models[model]
            self.emb_model = model_class.from_pretrained(model, output_hidden_states=True)
            self.tokenizer = tokenizer_class.from_pretrained(model)
        else:
            config = AutoConfig.from_pretrained(model, output_hidden_states=True)
            self.emb_model = AutoModel.from_pretrained(model, config=config)
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.emb_model.eval()
        self.emb_model.to(self.device)

    def get_embed_list(self, sent_batch) -> torch.Tensor:
        if self.emb_model is not None:
            with torch.no_grad():
                if not isinstance(sent_batch[0], str):
                    inputs = self.tokenizer(
                        sent_batch,
                        is_split_into_words=True,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                else:
                    inputs = self.tokenizer(
                        sent_batch,
                        is_split_into_words=False,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )

                hidden = self.emb_model(**inputs.to(self.device))["hidden_states"]
                if self.layer >= len(hidden):
                    raise ValueError(
                        f"Specified layer {self.layer}, but model has only {len(hidden)} layers."
                    )
                outputs = hidden[self.layer]
                return outputs[:, 1:-1, :]  # drop CLS & SEP
        return None


# -------------------------------
# Utility
# -------------------------------
def write_header_and_flush(path, n, dim):
    with open(path, "w", encoding="utf8") as out_text:
        out_text.write(f"{n} {dim}\n")


def append_lines(path, lines):
    with open(path, "a", encoding="utf8") as out_text:
        out_text.write("\n".join(lines) + "\n")


# -------------------------------
# XLM-R / Glot500 / Pretrained
# -------------------------------
def get_embedding(sentence, xlmr_embeddings):
    embedding_size = 768
    s_embedding = xlmr_embeddings.get_embed_list([[sentence]])
    assert s_embedding.size(dim=0) == 1, f"First dim is not 1: {s_embedding.size()}."
    if s_embedding.size(dim=1) > 1:
        np_embedding = s_embedding.cpu().detach().numpy()[0].mean(axis=0)
    elif s_embedding.size(dim=1) == 1:
        np_embedding = s_embedding.cpu().detach().numpy()[0][0]
    else:
        return None
    ls_embedding = np_embedding.tolist()
    assert len(ls_embedding) == embedding_size
    return [f"{x:.6f}" for x in ls_embedding]


def to_xlmr_sentence_embeddings(path, sentence_list, model_name, start_i=0):
    if model_name == "xlmr":
        model_name = "xlm-roberta-base"
    elif model_name == "glot500":
        model_name = "cis-lmu/glot500-base"
    elif model_name == "pretrained_partial":
        print(f"Using pretrained_partial model from: {PRETRAINING_PATH}")
        model_name = PRETRAINING_PATH
    elif model_name == "mmbert":
        model_name = "jhu-clsp/mmBERT-base"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xlmr_embeddings = EmbeddingLoader(model_name, device, layer=8)
    embedding_size = 768

    n = len(sentence_list)
    write_header_and_flush(path, n, embedding_size)

    batch_out = []
    for i in tqdm(range(start_i, len(sentence_list))):
        line = sentence_list[i]
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        sent_id = parts[0]
        sent_text = "\t".join(parts[1:])

        str_embedding = get_embedding(sent_text, xlmr_embeddings)
        if str_embedding:
            batch_out.append(f"{sent_id} {' '.join(str_embedding)}")

        if (i + 1) % 10000 == 0:
            append_lines(path, batch_out)
            batch_out = []

    if batch_out:
        append_lines(path, batch_out)


# -------------------------------
# LaBSE
# -------------------------------
def get_labse_embeddings(sentence, labse_model):
    emb = labse_model.encode(sentence)
    np_emb = emb.tolist()
    return [f"{x:.6f}" for x in np_emb]


def to_labse_sentence_embeddings(path, sentence_list, start_i=0):
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    embedding_size = 768

    n = len(sentence_list)
    write_header_and_flush(path, n, embedding_size)

    batch_out = []
    for i in tqdm(range(start_i, len(sentence_list))):
        line = sentence_list[i]
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        sent_id = parts[0]
        sent_text = "\t".join(parts[1:])

        str_embedding = get_labse_embeddings(sent_text, labse_model)
        if str_embedding:
            batch_out.append(f"{sent_id} {' '.join(str_embedding)}")

        if (i + 1) % 10000 == 0:
            append_lines(path, batch_out)
            batch_out = []

    if batch_out:
        append_lines(path, batch_out)


# -------------------------------
# SONAR
# -------------------------------
def to_sonar_sentence_embeddings(path, sentence_list, start_i=0):
    if not HAS_SONAR:
        raise ImportError("SONAR is not installed (sonar-space). Install it first.")

    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device,
    )

    # figure out dimension
    test_emb = pipe.predict(["hello"], source_lang="deu_Latn")
    dim = int(test_emb.shape[-1])

    n = len(sentence_list)
    with open(path, "w", encoding="utf8") as f:
        f.write(f"{n} {dim}\n")

    # split ids and texts
    ids, texts = [], []
    for line in sentence_list:
        parts = line.split("\t", 1)
        if len(parts) == 2:
            ids.append(parts[0])
            texts.append(parts[1])
        else:
            ids.append(parts[0])
            texts.append("")

    BATCH_SIZE = 64
    with open(path, "a", encoding="utf8") as out_f:
        for start in tqdm(range(0, len(texts), BATCH_SIZE), desc="SONAR embed"):
            end = start + BATCH_SIZE
            batch_ids = ids[start:end]
            batch_txt = texts[start:end]

            embs = pipe.predict(batch_txt, source_lang="deu_Latn").cpu().numpy()

            for sid, vec in zip(batch_ids, embs):
                out_f.write(
                    sid + " " + " ".join(f"{x:.6f}" for x in vec.tolist()) + "\n"
                )

QWEN3_MODEL_ID = "Qwen/Qwen3-Embedding-8B"

def to_qwen3_sentence_embeddings(path, sentence_list, start_i=0, batch_size=64, truncate_dim=None):
    """
    Writes .vec like the rest of your pipeline:
    first line: "N dim"
    then: "id v1 v2 ... vdim"
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use fp16 on GPU to reduce memory
    model_kwargs = {}
    if device == "cuda":
        model_kwargs = {"device_map": "auto", "torch_dtype": torch.float16}

    model = SentenceTransformer(
        QWEN3_MODEL_ID,
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )

    dim = int(model.get_sentence_embedding_dimension())
    if truncate_dim is not None:
        dim = int(truncate_dim)

    n = len(sentence_list)
    write_header_and_flush(path, n, dim)

    # split ids/texts
    ids, texts = [], []
    for line in sentence_list:
        line = line.rstrip("\n")
        parts = line.split("\t", 1)
        if len(parts) == 2:
            ids.append(parts[0])
            texts.append(parts[1])
        else:
            ids.append(parts[0])
            texts.append("")

    with open(path, "a", encoding="utf8") as out_f:
        for start in tqdm(range(0, len(texts), batch_size), desc="Qwen3 embed"):
            end = start + batch_size
            batch_ids = ids[start:end]
            batch_txt = texts[start:end]

            embs = model.encode(
                batch_txt,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )

            if truncate_dim is not None:
                embs = embs[:, :dim]

            for sid, vec in zip(batch_ids, embs):
                out_f.write(sid + " " + " ".join(f"{x:.6f}" for x in vec.tolist()) + "\n")
# -------------------------------
# LASER
# -------------------------------
def to_laser_sentence_embeddings(path, sentence_list, lang="deu", batch_size=64):
    """
    LASER2 version of the SONAR helper:
    - input:  path to write .vec file + list of lines "id<TAB>sentence"
    - output: text .vec file in the same format as SONAR: first line "N dim", then "id emb..."
    """
    try:
        from laser_encoders import LaserEncoderPipeline
    except ImportError as e:
        raise ImportError(
            "LASER is not installed (`laser_encoders`). Install it in this env."
        ) from e

    import numpy as np

    # init encoder (you tested that this works)
    enc = LaserEncoderPipeline(lang=lang)

    # probe dim
    test_emb = enc.encode_sentences(["hello"])
    dim = int(test_emb.shape[-1])

    # split ids / texts
    ids = []
    texts = []
    for line in sentence_list:
        line = line.rstrip("\n")
        parts = line.split("\t", 1)
        if len(parts) == 2:
            ids.append(parts[0])
            texts.append(parts[1])
        else:
            # fallback: line only has an id
            ids.append(parts[0])
            texts.append("")

    n = len(texts)

    # write header like the rest of your pipeline expects
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{n} {dim}\n")

    # optional progress bar
    try:
        from tqdm import tqdm
        rng = tqdm(range(0, n, batch_size), desc="LASER embed")
    except Exception:
        rng = range(0, n, batch_size)

    with open(path, "a", encoding="utf-8") as out_f:
        for start in rng:
            end = start + batch_size
            batch_ids = ids[start:end]
            batch_txt = texts[start:end]

            # LASER2 encode
            embs = enc.encode_sentences(batch_txt)  # (B, dim)
            embs = np.asarray(embs)

            for sid, vec in zip(batch_ids, embs):
                out_f.write(
                    sid + " " + " ".join(f"{x:.6f}" for x in vec.tolist()) + "\n"
                )
LLAMA31_MODEL_ID = "/dss/dsshome1/06/ge65hon2/models/llama-3.1-8b"

def to_llama31_lasttoken_sentence_embeddings(
    path,
    sentence_list,
    model_id=LLAMA31_MODEL_ID,
    batch_size=8,
    max_length=256,
    truncate_dim=768,   # 0 -> full dim
    hf_token=None,
    use_prompt=True,   # <<< NEW
):
    """
    Output .vec format:
      first line: "N dim"
      then: "id v1 v2 ... vdim"

    Embedding:
      - FINAL layer
      - last non-padding token
      - optional prompt:
        'Summarize sentence "{s}" in one word:'
    """

    import torch
    from transformers import AutoModel, AutoTokenizer
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Prompt template (Li & Li 2024)
    # -----------------------------
    def make_prompt(s):
        return f'Summarize sentence "{s}" in one word:'

    # -----------------------------
    # Load tokenizer + model
    # -----------------------------
    is_local_path = ("/" in model_id)

    tok = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=is_local_path,
        token=None if is_local_path else hf_token,
        use_fast=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = AutoModel.from_pretrained(
        model_id,
        local_files_only=is_local_path,
        token=None if is_local_path else hf_token,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.eval()

    # -----------------------------
    # Detect hidden dimension
    # -----------------------------
    full_dim = int(getattr(model.config, "hidden_size", 0) or getattr(model.config, "dim", 0))
    if full_dim <= 0:
        tmp = tok(["hello"], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            tmp_out = model(**tmp)
        full_dim = int(tmp_out.last_hidden_state.shape[-1])

    dim = full_dim if (truncate_dim is None or truncate_dim == 0) else int(truncate_dim)

    # -----------------------------
    # Split IDs and texts
    # -----------------------------
    ids, texts = [], []
    for line in sentence_list:
        line = line.rstrip("\n")
        parts = line.split("\t", 1)
        if len(parts) == 2:
            ids.append(parts[0])
            texts.append(parts[1])
        else:
            ids.append(parts[0])
            texts.append("")

    n = len(texts)
    write_header_and_flush(path, n, dim)

    # -----------------------------
    # Device for inputs
    # -----------------------------
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = device

    # -----------------------------
    # Main loop
    # -----------------------------
    with open(path, "a", encoding="utf8") as out_f:
        for start in tqdm(range(0, n, batch_size), desc="LLAMA31 last-token embed"):
            end = min(start + batch_size, n)

            batch_ids = ids[start:end]
            raw_txt = texts[start:end]

            if use_prompt:
                batch_txt = [make_prompt(s) for s in raw_txt]
            else:
                batch_txt = raw_txt

            inputs = tok(
                batch_txt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)
                h = out.last_hidden_state  # (B, T, D)

            attn = inputs["attention_mask"]          # (B, T)
            last_idx = attn.sum(dim=1) - 1            # (B,)

            emb = h[torch.arange(h.size(0), device=h.device), last_idx, :]
            emb = emb.to(torch.float32).cpu().numpy()

            if dim != full_dim:
                emb = emb[:, :dim]

            for sid, vec in zip(batch_ids, emb):
                out_f.write(
                    sid + " " + " ".join(f"{x:.6f}" for x in vec.tolist()) + "\n"
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

    if model_name in ["xlmr", "glot500", "pretrained_partial", "mmbert"]:
        to_xlmr_sentence_embeddings(args.output_file, split_file, model_name)
    elif model_name == "labse":
        to_labse_sentence_embeddings(args.output_file, split_file)
    elif model_name == "sonar":
        to_sonar_sentence_embeddings(args.output_file, split_file)
    elif model_name == "qwen3":
        to_qwen3_sentence_embeddings(args.output_file, split_file)
    elif model_name == "llama31_noprompt":
            to_llama31_lasttoken_sentence_embeddings(
                    args.output_file,
                    split_file,
                    batch_size = args.batch_size,
                max_length = args.max_length,
                truncate_dim = args.truncate_dim,
                hf_token = args.hf_token,)
    elif model_name == "laser":
        # default lang="deu"; change if your input is another lang
        to_laser_sentence_embeddings(args.output_file, split_file, lang="deu")
    else:
        raise ValueError(f"Unknown model {model_name}")

    print(f"Done. Saved embeddings to {args.output_file}")


if __name__ == "__main__":
    main()
