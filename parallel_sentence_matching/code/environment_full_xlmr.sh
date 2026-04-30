#!/usr/bin/env bash
set -x
set -e

THIS_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "$THIS_DIR/.." && pwd)"

PYTHON=python
FASTTEXT="$PROJECT_ROOT/code/third_party/fastText/fasttext"
MUSE="$PROJECT_ROOT/code/third_party/MUSE"
MOSES="$PROJECT_ROOT/code/third_party/moses/scripts"

DATA="${DATA:-$PROJECT_ROOT/data}"

# ================= PIVOT SETTINGS ===========================================
# Change this ONE default if you want to hard-code a different pivot:
PIVOT_TAG="${PIVOT_TAG:-deu_Latn}"

# ================= LANGUAGE SETTINGS (defaults; mine.sh overwrites in custom) =
SRC_LANGS='hsb'
TRG_LANGS='de'
BUCC_SETS='train test'

# ================= MODEL SETTINGS ===========================================
BACKEND="${BACKEND:-pretrained}"
PREFIX="${BACKEND}."

# ================= RESULTS & OUTPUT FOLDERS =================================
# Pivot-aware root:
RESULTS_ROOT="$PROJECT_ROOT/results/${PIVOT_TAG}"

RAW_RESULTS_ROOT="$RESULTS_ROOT/raw/$BACKEND"
CBIE_RESULTS_ROOT="$RESULTS_ROOT/cbie/$BACKEND"
WHITENED_RESULTS_ROOT="$RESULTS_ROOT/whitened/$BACKEND"

# Default points to RAW (mine.sh will override depending on CBIE/WHITENED)
RESULTS="$RAW_RESULTS_ROOT"

EMBEDDINGS="${RESULTS}/embeddings"
DOC_EMBEDDINGS="${EMBEDDINGS}/doc"
DICTIONARIES="${RESULTS}/dictionaries"
MINING="${RESULTS}/mining"
FILTERING="${RESULTS}/filtering"

mkdir -p "$DOC_EMBEDDINGS" "$DICTIONARIES" "$MINING" "$FILTERING"

# ================= MINING PARAMETERS ========================================
THREADS=20
GPUS=0
DIM=768
TOPN_DICT=100
TOPN_CSLS=500
TOPN_DOC=100

export PROJECT_ROOT DATA RESULTS RESULTS_ROOT RAW_RESULTS_ROOT CBIE_RESULTS_ROOT WHITENED_RESULTS_ROOT \
       EMBEDDINGS DOC_EMBEDDINGS DICTIONARIES MINING FILTERING \
       BACKEND PREFIX THREADS GPUS DIM TOPN_DICT TOPN_CSLS TOPN_DOC \
       PYTHON FASTTEXT MUSE MOSES PIVOT_TAG
