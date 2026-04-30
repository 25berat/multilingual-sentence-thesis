import os
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
# 🔧 CONFIG (EDIT HERE ONLY)
# =========================
TRAIN_TXT = r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\train\merged_shuffled.txt"
OUT_DIR = r"C:\Users\berat\PycharmProjects\PaSeMiLL\code\model\mmbert_ft"
MODEL_ID = "jhu-clsp/mmBERT-base"

MAX_LENGTH = 128
BATCH_SIZE = 8            # safer default
GRAD_ACCUM = 4           # effective batch = 32
EPOCHS = 1
LR = 5e-5
MLM_PROB = 0.15
NUM_WORKERS = 2          # reduce if Windows issues

# =========================
# 🚀 MAIN
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID)

    print("Loading dataset...")
    dataset = load_dataset("text", data_files={"train": TRAIN_TXT})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=NUM_WORKERS,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    fp16 = torch.cuda.is_available()

    print("Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        overwrite_output_dir=True,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        logging_steps=200,
        save_steps=2000,
        save_total_limit=2,
        fp16=fp16,
        dataloader_num_workers=0,   # safer on Windows
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    print("🚀 Training started...")
    trainer.train()

    print("💾 Saving model...")
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print("\n✅ Done!")
    print(f"Model saved at: {OUT_DIR}")


if __name__ == "__main__":
    main()