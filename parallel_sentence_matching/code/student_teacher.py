import os
import sys
import json
import logging
import traceback
import faulthandler
from datetime import datetime

import numpy as np
import torch
from datasets import DatasetDict, load_dataset, get_dataset_config_names
from datasets.utils.logging import disable_progress_bar

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import (
    EmbeddingSimilarityEvaluator,
    MSEEvaluator,
    SequentialEvaluator,
    TranslationEvaluator,
)
from sentence_transformers.losses import MSELoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


# =========================
# PATHS / ENV
# =========================
BASE_WORKDIR = os.environ.get(
    "BASE_WORKDIR",
    "/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge65hon2"
)

RUN_DIR = f"{BASE_WORKDIR}/student_teacher_run_debug"
HF_CACHE_DIR = os.environ.get("HF_HOME", f"{BASE_WORKDIR}/hf_cache")
HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", f"{BASE_WORKDIR}/hf_datasets_cache")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", f"{BASE_WORKDIR}/hf_hub_cache")
TMP_DIR = os.environ.get("TMPDIR", f"{BASE_WORKDIR}/tmp")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
os.makedirs(HF_HUB_CACHE, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_HUB_CACHE
os.environ["TMPDIR"] = TMP_DIR
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

disable_progress_bar()

RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_PATH = f"{RUN_DIR}/run_{RUN_ID}"
os.makedirs(RUN_PATH, exist_ok=True)


# =========================
# FAULTHANDLER
# =========================
faulthandler_log_path = f"{RUN_PATH}/faulthandler.log"
faulthandler_file = open(faulthandler_log_path, "w")
faulthandler.enable(faulthandler_file)


# =========================
# LOGGING
# =========================
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"{RUN_PATH}/run.log"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


# =========================
# CONFIG
# =========================
teacher_model_name = "LaBSE"
student_model_name = "cis-lmu/glot500-base"

student_max_seq_length = 128
train_batch_size = 16
inference_batch_size = 64
max_sentences_per_language = 100_000
preprocess_batch_size = 2048

num_train_epochs = 1
num_evaluation_steps = 5000

source_languages = ["en"]
target_languages = [
    "de", "ar", "bg", "ca", "cs", "da", "el", "es", "et", "fa", "fi", "fr",
    "fr-ca", "gl", "gu", "he", "hi", "hr", "hu", "hy", "id", "it", "ja", "ka",
    "ko", "ku", "lt", "lv", "mk", "mn", "mr", "ms", "my", "nb", "nl", "pl",
    "pt-br", "pt", "ro", "ru", "sk", "sl", "sq", "sr", "sv", "th", "tr", "uk",
    "ur", "vi", "zh-cn", "zh-tw",
]

dataset_to_use = "sentence-transformers/parallel-sentences-talks"
sts_dataset_name = "mteb/sts17-crosslingual-sts"

debug_subsets_env = os.environ.get("DEBUG_SUBSETS", "").strip()
debug_subsets = set(x.strip() for x in debug_subsets_env.split(",") if x.strip())

force_redownload = os.environ.get("FORCE_REDOWNLOAD", "0") == "1"
download_mode = "force_redownload" if force_redownload else None

output_dir = f"{RUN_PATH}/model"
os.makedirs(output_dir, exist_ok=True)


# =========================
# HELPERS
# =========================
def json_dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def log_exception(context, subset=None, stage=None):
    err = traceback.format_exc()
    logger.error(context)
    logger.error(err)
    append_jsonl(
        f"{RUN_PATH}/subset_failures.jsonl",
        {
            "time": datetime.now().isoformat(),
            "subset": subset,
            "stage": stage,
            "context": context,
            "traceback": err,
        },
    )


# =========================
# RUN METADATA
# =========================
run_metadata = {
    "run_id": RUN_ID,
    "run_path": RUN_PATH,
    "base_workdir": BASE_WORKDIR,
    "dataset_to_use": dataset_to_use,
    "sts_dataset_name": sts_dataset_name,
    "teacher_model_name": teacher_model_name,
    "student_model_name": student_model_name,
    "student_max_seq_length": student_max_seq_length,
    "train_batch_size": train_batch_size,
    "inference_batch_size": inference_batch_size,
    "preprocess_batch_size": preprocess_batch_size,
    "max_sentences_per_language": max_sentences_per_language,
    "num_train_epochs": num_train_epochs,
    "num_evaluation_steps": num_evaluation_steps,
    "source_languages": source_languages,
    "target_languages": target_languages,
    "debug_subsets": sorted(debug_subsets),
    "force_redownload": force_redownload,
    "env": {
        "HF_HOME": os.environ.get("HF_HOME"),
        "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE"),
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "HF_HUB_DISABLE_PROGRESS_BARS": os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),
    },
}
json_dump(run_metadata, f"{RUN_PATH}/run_metadata.json")


# =========================
# DEVICE INFO
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

logger.info(f"BASE_WORKDIR={BASE_WORKDIR}")
logger.info(f"HF_HOME={os.environ['HF_HOME']}")
logger.info(f"HF_DATASETS_CACHE={os.environ['HF_DATASETS_CACHE']}")
logger.info(f"HUGGINGFACE_HUB_CACHE={os.environ['HUGGINGFACE_HUB_CACHE']}")
logger.info(f"HF_HUB_CACHE={os.environ['HF_HUB_CACHE']}")
logger.info(f"TMPDIR={os.environ['TMPDIR']}")
logger.info(f"RUN_PATH={RUN_PATH}")
logger.info(f"DEBUG_SUBSETS={sorted(debug_subsets) if debug_subsets else 'ALL'}")
logger.info(f"FORCE_REDOWNLOAD={force_redownload}")


# =========================
# VALID CONFIGS CHECK
# =========================
logger.info(f"Fetching dataset configs for {dataset_to_use}")
valid_configs = set(get_dataset_config_names(dataset_to_use))
logger.info(f"Found {len(valid_configs)} valid subsets")

json_dump(sorted(valid_configs), f"{RUN_PATH}/valid_configs.json")


# =========================
# LOAD MODELS
# =========================
logger.info("Loading teacher model...")
teacher_model = SentenceTransformer(teacher_model_name, device=device)
logger.info(f"Teacher model loaded: {teacher_model_name}")

logger.info("Loading student model...")
student_model = SentenceTransformer(student_model_name, device=device)
student_model.max_seq_length = student_max_seq_length
logger.info(f"Student model loaded: {student_model_name}")
logger.info(f"Student max_seq_length set to {student_model.max_seq_length}")


# =========================
# LOAD DATASETS
# =========================
train_dataset_dict = DatasetDict()
eval_dataset_dict = DatasetDict()

dataset_summary = {
    "loaded_train_subsets": [],
    "loaded_eval_subsets": [],
    "skipped_not_in_configs": [],
    "failed_train": [],
    "failed_dev": [],
    "failed_preprocess_train": [],
    "failed_preprocess_eval": [],
    "sts_loaded": [],
    "sts_missing": [],
}

for source_lang in source_languages:
    for target_lang in target_languages:
        subset = f"{source_lang}-{target_lang}"

        if debug_subsets and subset not in debug_subsets:
            logger.info(f"Skipping {subset} because DEBUG_SUBSETS is active")
            continue

        logger.info("=" * 80)
        logger.info(f"Checking subset: {subset}")

        if subset not in valid_configs:
            logger.warning(f"Skipping {subset} (not in dataset configs)")
            dataset_summary["skipped_not_in_configs"].append(subset)
            continue

        try:
            logger.info(f"Loading TRAIN for {subset}")
            train_dataset = load_dataset(
                dataset_to_use,
                subset,
                split="train",
                download_mode=download_mode,
            )
            logger.info(f"TRAIN loaded for {subset}: {len(train_dataset)} rows")

            if len(train_dataset) > max_sentences_per_language:
                logger.info(
                    f"Reducing train dataset for {subset} from {len(train_dataset)} "
                    f"to {max_sentences_per_language}"
                )
                train_dataset = train_dataset.select(range(max_sentences_per_language))
        except Exception:
            log_exception(f"FAILED TRAIN for {subset}", subset=subset, stage="train")
            dataset_summary["failed_train"].append(subset)
            continue

        try:
            logger.info(f"Loading DEV for {subset}")
            eval_dataset = load_dataset(
                dataset_to_use,
                subset,
                split="dev",
                download_mode=download_mode,
            )
            logger.info(f"DEV loaded for {subset}: {len(eval_dataset)} rows")

            if len(eval_dataset) > 1000:
                logger.info(f"Reducing dev dataset for {subset} from {len(eval_dataset)} to 1000")
                eval_dataset = eval_dataset.select(range(1000))
        except Exception:
            logger.warning(f"No usable dev split for {subset}, trying fallback split from train")
            try:
                split_size = min(1000, max(1, len(train_dataset) // 10))
                split = train_dataset.train_test_split(test_size=split_size, shuffle=True)
                train_dataset = split["train"]
                eval_dataset = split["test"]
                logger.info(
                    f"Fallback split created for {subset}: "
                    f"train={len(train_dataset)} eval={len(eval_dataset)}"
                )
            except Exception:
                log_exception(f"FAILED DEV fallback for {subset}", subset=subset, stage="dev")
                dataset_summary["failed_dev"].append(subset)
                continue

        train_dataset_dict[subset] = train_dataset
        eval_dataset_dict[subset] = eval_dataset
        dataset_summary["loaded_train_subsets"].append(subset)
        dataset_summary["loaded_eval_subsets"].append(subset)

        logger.info(f"SUCCESS {subset} | train={len(train_dataset)} | eval={len(eval_dataset)}")

logger.info(f"Loaded train subsets: {list(train_dataset_dict.keys())}")
logger.info(f"Loaded eval subsets: {list(eval_dataset_dict.keys())}")

json_dump(dataset_summary, f"{RUN_PATH}/dataset_load_summary.json")

if len(train_dataset_dict) == 0:
    raise ValueError("No training datasets were loaded successfully.")


# =========================
# PREPARE DATA
# =========================
def prepare_dataset(batch):
    labels = teacher_model.encode(
        batch["english"],
        batch_size=inference_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return {
        "english": batch["english"],
        "non_english": batch["non_english"],
        "label": labels,
    }


logger.info("Starting preprocessing for train/eval datasets...")

for subset in list(train_dataset_dict.keys()):
    logger.info("-" * 80)
    logger.info(f"Preprocessing TRAIN subset: {subset}")
    try:
        train_dataset_dict[subset] = train_dataset_dict[subset].map(
            prepare_dataset,
            batched=True,
            batch_size=preprocess_batch_size,
            desc=f"Preprocess train {subset}",
        )
        logger.info(f"TRAIN preprocessing done for {subset}")
    except Exception:
        log_exception(f"PREPROCESS TRAIN FAIL for {subset}", subset=subset, stage="preprocess_train")
        dataset_summary["failed_preprocess_train"].append(subset)
        del train_dataset_dict[subset]
        if subset in eval_dataset_dict:
            del eval_dataset_dict[subset]
        continue

    logger.info(f"Preprocessing EVAL subset: {subset}")
    try:
        eval_dataset_dict[subset] = eval_dataset_dict[subset].map(
            prepare_dataset,
            batched=True,
            batch_size=preprocess_batch_size,
            desc=f"Preprocess eval {subset}",
        )
        logger.info(f"EVAL preprocessing done for {subset}")
    except Exception:
        log_exception(f"PREPROCESS EVAL FAIL for {subset}", subset=subset, stage="preprocess_eval")
        dataset_summary["failed_preprocess_eval"].append(subset)
        del train_dataset_dict[subset]
        del eval_dataset_dict[subset]
        continue

json_dump(dataset_summary, f"{RUN_PATH}/dataset_load_summary.json")

if len(train_dataset_dict) == 0:
    raise ValueError("All datasets failed during preprocessing.")


# =========================
# DEFINE LOSS
# =========================
train_loss = MSELoss(model=student_model)


# =========================
# DEFINE EVALUATORS
# =========================
logger.info("Creating evaluators...")
evaluators = []

for subset, eval_dataset in eval_dataset_dict.items():
    logger.info(f"Creating evaluators for {subset}")

    try:
        dev_mse = MSEEvaluator(
            source_sentences=eval_dataset["english"],
            target_sentences=eval_dataset["non_english"],
            name=subset,
            teacher_model=teacher_model,
            batch_size=inference_batch_size,
        )
        evaluators.append(dev_mse)

        dev_trans_acc = TranslationEvaluator(
            source_sentences=eval_dataset["english"],
            target_sentences=eval_dataset["non_english"],
            name=subset,
            batch_size=inference_batch_size,
        )
        evaluators.append(dev_trans_acc)
    except Exception:
        log_exception(f"EVALUATOR CREATION FAIL for {subset}", subset=subset, stage="evaluator")
        continue

    test_dataset = None
    sts_subset_name = subset

    try:
        test_dataset = load_dataset(
            sts_dataset_name,
            subset,
            split="test",
            download_mode=download_mode,
        )
    except Exception:
        try:
            reversed_subset = f"{subset[3:]}-{subset[:2]}"
            test_dataset = load_dataset(
                sts_dataset_name,
                reversed_subset,
                split="test",
                download_mode=download_mode,
            )
            sts_subset_name = reversed_subset
        except Exception:
            test_dataset = None

    if test_dataset is not None:
        try:
            test_evaluator = EmbeddingSimilarityEvaluator(
                sentences1=test_dataset["sentence1"],
                sentences2=test_dataset["sentence2"],
                scores=[score / 5.0 for score in test_dataset["score"]],
                batch_size=inference_batch_size,
                name=f"sts17-{sts_subset_name}-test",
                show_progress_bar=False,
            )
            evaluators.append(test_evaluator)
            dataset_summary["sts_loaded"].append(sts_subset_name)
        except Exception:
            log_exception(
                f"STS EVALUATOR FAIL for {sts_subset_name}",
                subset=sts_subset_name,
                stage="sts_evaluator",
            )
    else:
        dataset_summary["sts_missing"].append(subset)

json_dump(dataset_summary, f"{RUN_PATH}/dataset_load_summary.json")

if len(evaluators) == 0:
    logger.warning("No evaluators were created. Training will continue without evaluator.")
    evaluator = None
else:
    evaluator = SequentialEvaluator(
        evaluators,
        main_score_function=lambda scores: np.mean(scores),
    )


# =========================
# TRAINING ARGS
# =========================
logger.info("Building training arguments...")
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available(),
    bf16=False,
    learning_rate=2e-5,
    eval_strategy="steps" if evaluator is not None else "no",
    eval_steps=num_evaluation_steps if evaluator is not None else None,
    save_strategy="steps",
    save_steps=num_evaluation_steps,
    save_total_limit=2,
    logging_steps=100,
    run_name=f"multilingual-{'-'.join(source_languages)}-debug",
)


# =========================
# TRAINER
# =========================
logger.info("Creating trainer...")
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset_dict,
    eval_dataset=eval_dataset_dict if evaluator is not None else None,
    loss=train_loss,
    evaluator=evaluator,
)


# =========================
# TRAIN
# =========================
logger.info("Starting training...")
try:
    trainer.train()
    logger.info("Training finished successfully")
except Exception:
    log_exception("TRAINING FAILED", stage="training")
    raise


# =========================
# SAVE FINAL MODEL
# =========================
final_output_dir = f"{output_dir}/final"
os.makedirs(final_output_dir, exist_ok=True)

try:
    student_model.save(final_output_dir)
    logger.info(f"Final model saved to: {final_output_dir}")
except Exception:
    log_exception("FINAL MODEL SAVE FAILED", stage="save")
    raise


# =========================
# FINAL SUMMARY
# =========================
final_summary = {
    "run_id": RUN_ID,
    "run_path": RUN_PATH,
    "final_output_dir": final_output_dir,
    "loaded_train_subsets_count": len(dataset_summary["loaded_train_subsets"]),
    "loaded_eval_subsets_count": len(dataset_summary["loaded_eval_subsets"]),
    "failed_train_count": len(dataset_summary["failed_train"]),
    "failed_dev_count": len(dataset_summary["failed_dev"]),
    "failed_preprocess_train_count": len(dataset_summary["failed_preprocess_train"]),
    "failed_preprocess_eval_count": len(dataset_summary["failed_preprocess_eval"]),
    "sts_loaded_count": len(dataset_summary["sts_loaded"]),
    "sts_missing_count": len(dataset_summary["sts_missing"]),
}
json_dump(final_summary, f"{RUN_PATH}/final_summary.json")
json_dump(dataset_summary, f"{RUN_PATH}/dataset_load_summary.json")

logger.info("Run completed.")
logger.info(f"Final summary saved to: {RUN_PATH}/final_summary.json")

faulthandler_file.close()
