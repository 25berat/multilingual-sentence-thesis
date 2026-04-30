#!/usr/bin/env bash
set -euo pipefail

########################################
# Small wrapper: run main pipeline
# for multiple BACKEND models
########################################

# Folder of this script
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# Main mining script (the big one you pasted)
MAIN_SCRIPT="${SCRIPT_DIR}/mine_bucc_full_xlmr.sh"

# Models as expected by contextual_sentence_embeddings.py: -m <model_name>
# Comment out ones you don't want yet (e.g. sonar / laser if not installed).
MODELS=(
  "glot500"
  "labse"
  "laser"
  "sonar"
  "xlmr"
  )

# Respect existing toggles if you export them before calling this script
USE_CUSTOM_DATA="${USE_CUSTOM_DATA:-1}"
CBIE="${CBIE:-0}"
WHITENED="${WHITENED:-1}"

export USE_CUSTOM_DATA CBIE WHITENED

for model in "${MODELS[@]}"; do
  echo
  echo "============================================="
  echo "  Running mining pipeline for model: $model"
  echo "  (USE_CUSTOM_DATA=$USE_CUSTOM_DATA, CBIE=$CBIE, WHITENED=$WHITENED)"
  echo "============================================="
  echo

  # Set BACKEND for the inner script + environment_full_xlmr.sh
  export BACKEND="$model"

  # (optional) separate result trees per model, if you like:
  # export RAW_RESULTS_ROOT="${PROJECT_ROOT}/results/${model}/raw"
  # export CBIE_RESULTS_ROOT="${PROJECT_ROOT}/results/${model}/cbie"
  # export WHITENED_RESULTS_ROOT="${PROJECT_ROOT}/results/${model}/whitened"

  bash "$MAIN_SCRIPT"
done
