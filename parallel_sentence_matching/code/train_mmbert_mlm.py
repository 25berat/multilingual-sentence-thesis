#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import inspect
import signal
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# =========================
# CONFIG
# =========================
TRAIN_TXT = "/dss/dsshome1/06/ge65hon2/projects/PaSeMiLL/data/train/merged_shuffled.txt"
OUT_DIR   = "/dss/dsshome1/06/ge65hon2/projects/PaSeMiLL/code/model/mmbert_ft_finalonly"
MODEL_ID  = "jhu-clsp/mmBERT-base"

MAX_LENGTH = 128
BATCH_SIZE = 12
GRAD_ACCUM = 2
LR = 5e-5
MLM_PROB = 0.15

LOGGING_STEPS = 200
DATALOADER_WORKERS = 0

# leichtes Mischen im Stream
SHUFFLE_BUFFER = 0

# map batch size für streaming tokenization
MAP_BATCH_SIZE = 2000


# =========================
# Helpers
# =========================
def supported_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def count_nonempty_lines(path: str) -> int:
    print(f"Counting non-empty lines in: {path}", flush=True)
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tokenizer & model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)

    # tied weights warning vermeiden
    if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

    # -------------------------
    # count lines -> approximate 1 epoch over file
    # -------------------------
    num_examples = count_nonempty_lines(TRAIN_TXT)
    effective_batch_size = BATCH_SIZE * GRAD_ACCUM
    max_steps = math.ceil(num_examples / effective_batch_size)

    print(f"Non-empty lines:        {num_examples:,}", flush=True)
    print(f"Per-device batch size:  {BATCH_SIZE}", flush=True)
    print(f"Grad accumulation:      {GRAD_ACCUM}", flush=True)
    print(f"Effective batch size:   {effective_batch_size}", flush=True)
    print(f"Training steps (~1 pass over file): {max_steps:,}", flush=True)

    print("Loading dataset (streaming)...", flush=True)
    ds = load_dataset("text", data_files={"train": TRAIN_TXT}, streaming=True)
    train_stream = ds["train"]

    if SHUFFLE_BUFFER and SHUFFLE_BUFFER > 0:
        print(f"Applying streaming shuffle with buffer={SHUFFLE_BUFFER:,}", flush=True)
        train_stream = train_stream.shuffle(buffer_size=SHUFFLE_BUFFER, seed=42)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Preparing streaming tokenization...", flush=True)
    tokenized_train = train_stream.map(
        tokenize,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    use_fp16 = torch.cuda.is_available()

    print("Setting up training...", flush=True)

    base_kwargs = dict(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        max_steps=max_steps,
        logging_steps=LOGGING_STEPS,
        fp16=use_fp16,
        gradient_checkpointing=False,
        dataloader_num_workers=DATALOADER_WORKERS,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",

        # no checkpoints during training
        save_strategy="no",
        save_steps=10**12,
        save_total_limit=1,
    )

    training_args = TrainingArguments(**supported_training_args(**base_kwargs))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    # -------------------------
    # save on SLURM timeout/cancel
    # -------------------------
    _saved = {"done": False}

    def _final_save_and_exit(signum, frame):
        if _saved["done"]:
            sys.exit(0)
        _saved["done"] = True

        sig_name = {
            getattr(signal, "SIGUSR1", None): "SIGUSR1",
            getattr(signal, "SIGTERM", None): "SIGTERM",
        }.get(signum, str(signum))

        print(f"\nReceived {sig_name}. Saving final model now...", flush=True)

        try:
            trainer.save_model(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            print("Final model saved successfully.", flush=True)
        except Exception as e:
            print(f"Final save failed: {e}", flush=True)
        finally:
            sys.exit(0)

    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _final_save_and_exit)

    signal.signal(signal.SIGTERM, _final_save_and_exit)

    # -------------------------
    # train
    # -------------------------
    print("Training started...", flush=True)
    trainer.train()

    print("Saving final model...", flush=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("\nDone!", flush=True)
    print(f"Model saved at: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
