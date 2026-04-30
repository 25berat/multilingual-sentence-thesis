#!/usr/bin/env bash
set -euo pipefail

############################################
# TOGGLES
############################################
USE_CUSTOM_DATA=${USE_CUSTOM_DATA:-1}
CBIE=${CBIE:-0}         # 0 = RAW run (compute embeddings); 1 = use CBIE vecs only
WHITENED=${WHITENED:-0} # 1 = use WHITENED vecs only

# RAW: run pipeline ONLY when embeddings missing
# CBIE/WHITENED: run mining ONLY when mining outputs missing
PIPELINE_ONLY_ON_EMB_MISSING=${PIPELINE_ONLY_ON_EMB_MISSING:-0}

############################################
# Load environment (sets PROJECT_ROOT, RAW/CBIE/WHITENED roots, BACKEND, etc.)
############################################
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/environment_full_xlmr.sh"

############################################
# Decide which result tree we operate on
############################################
if [[ "$WHITENED" == "1" ]]; then
  echo "[mode] WHITENED mode enabled → using $WHITENED_RESULTS_ROOT"
  RESULTS="$WHITENED_RESULTS_ROOT"
elif [[ "$CBIE" == "1" ]]; then
  echo "[mode] CBIE mode enabled → using $CBIE_RESULTS_ROOT"
  RESULTS="$CBIE_RESULTS_ROOT"
else
  echo "[mode] RAW mode → using $RAW_RESULTS_ROOT"
  RESULTS="$RAW_RESULTS_ROOT"
fi

# derive subfolders from chosen RESULTS
DOC_EMBEDDINGS="${RESULTS}/embeddings/doc"
DICTIONARIES="${RESULTS}/dictionaries"
MINING="${RESULTS}/mining"
FILTERING="${RESULTS}/filtering"
TMP_IN="${RESULTS}/tmp_inputs"

mkdir -p "$DOC_EMBEDDINGS" "$DICTIONARIES" "$MINING" "$FILTERING" "$TMP_IN"

############################################
# Local TXT plug-in defaults
############################################
RAW_TXT_DIR=${RAW_TXT_DIR:-"$PROJECT_ROOT/data/unshuffled_custom_data"}   # pivot base (for pivot embedding)
SHUF_TXT_DIR=${SHUF_TXT_DIR:-"$PROJECT_ROOT/data/shuffled_custom_data"}   # shuffled texts
PIVOT_TAG=${PIVOT_TAG:-"deu_Latn"}                                        # single pivot variable

if [[ "$USE_CUSTOM_DATA" == "1" ]]; then
  echo "[plug-in] Using local TXT data"
  echo "[plug-in] RAW_TXT_DIR : $RAW_TXT_DIR"
  echo "[plug-in] SHUF_TXT_DIR: $SHUF_TXT_DIR"
  echo "[plug-in] PIVOT_TAG   : $PIVOT_TAG"

  LANGS=()
  shopt -s nullglob
  for f in "$SHUF_TXT_DIR"/*.txt; do
    b=$(basename "$f" .txt)
    if [[ "$b" =~ ^[a-z]{3,}_[A-Za-z]+$ && "$b" != "$PIVOT_TAG" ]]; then
      LANGS+=("$b")
    fi
  done
  shopt -u nullglob

  SRC_LANGS="$PIVOT_TAG"
  TRG_LANGS="${LANGS[*]}"
  BUCC_SETS="custom"

  echo "[plug-in] SRC_LANGS: $SRC_LANGS"
  echo "[plug-in] TRG_LANGS: $TRG_LANGS"
  echo "[plug-in] BUCC_SETS: $BUCC_SETS"
fi

# ensure pair folders
for src_lang in $SRC_LANGS; do
  for trg_lang in $TRG_LANGS; do
    pair="$src_lang-$trg_lang"
    mkdir -p "${DOC_EMBEDDINGS}/bucc2017/${pair}"
    mkdir -p "${DICTIONARIES}/bucc2017/${pair}"
    mkdir -p "${MINING}/bucc2017/${pair}"
    mkdir -p "${FILTERING}/bucc2017/${pair}"
  done
done

# -----------------------------------------
# helper: text -> numeric-id tsv
# -----------------------------------------
ensure_tsv () {
  local src="$1"
  local dst="$2"

  if command -v dos2unix >/dev/null 2>&1; then
    dos2unix -q "$src" 2>/dev/null || true
  else
    sed $'s/\r$//' "$src" > "${src}.nocr" && mv "${src}.nocr" "$src"
  fi

  awk 'BEGIN{i=0} length($0)>0 {printf "%d\t%s\n", i, $0; i++}' "$src" > "$dst"
}

for data in $BUCC_SETS; do

  ####################################################################
  # 1) PIVOT EMBEDDING (computed once per run in RAW mode)
  # RAW mode → create
  # CBIE/WHITENED mode → expect it to exist in this RESULTS tree
  ####################################################################
  GLOBAL_PIVOT_DIR="${DOC_EMBEDDINGS}/global_${PIVOT_TAG}"
  mkdir -p "$GLOBAL_PIVOT_DIR"
  GLOBAL_PIVOT_VEC="${GLOBAL_PIVOT_DIR}/${PREFIX}${PIVOT_TAG}.${data}.${PIVOT_TAG}.vec"

  if [[ "$WHITENED" == "0" && "$CBIE" == "0" ]]; then
    if [[ ! -s "$GLOBAL_PIVOT_VEC" ]]; then
      echo "[pivot emb] computing once → $GLOBAL_PIVOT_VEC"
      base_in="$RAW_TXT_DIR/$PIVOT_TAG.txt"
      if [[ ! -f "$base_in" ]]; then
        echo "[error] pivot txt not found: $base_in"
        exit 1
      fi

      tmp_pivot_tsv="$TMP_IN/${PIVOT_TAG}.${data}.tsv"
      ensure_tsv "$base_in" "$tmp_pivot_tsv"

      $PYTHON contextual_sentence_embeddings.py \
        --input_file "$tmp_pivot_tsv" \
        --output_file "$GLOBAL_PIVOT_VEC" \
        -m "${BACKEND}"
    else
      echo "[pivot emb] reusing existing $GLOBAL_PIVOT_VEC"
    fi
  else
    if [[ ! -s "$GLOBAL_PIVOT_VEC" ]]; then
      mode="cbie"
      if [[ "$WHITENED" == "1" ]]; then mode="whitened"; fi
      echo "[warn][$mode] expected ${GLOBAL_PIVOT_VEC} to exist but it's missing"
    fi
  fi

  ####################################################################
  # 2) PAIRS
  ####################################################################
  for src_lang in $SRC_LANGS; do
    for trg_lang in $TRG_LANGS; do
      pair="$src_lang-$trg_lang"
      pair_embed_dir="${DOC_EMBEDDINGS}/bucc2017/${pair}"
      pair_dict_dir="${DICTIONARIES}/bucc2017/${pair}"
      pair_mining_dir="${MINING}/bucc2017/${pair}"
      pair_filter_dir="${FILTERING}/bucc2017/${pair}"

      mkdir -p "$pair_embed_dir" "$pair_dict_dir" "$pair_mining_dir" "$pair_filter_dir"

      # Names (keep consistent everywhere)
      sim_out="${pair_dict_dir}/DOC.${PREFIX}.${pair}.${data}.sim"
      pred="${pair_mining_dir}/${PREFIX}${data}.sim.pred"
      res_file="${pred}.res"

      ############################################################
      # RAW ONLY: skip FULL pipeline if embeddings already exist
      ############################################################
      if [[ "$PIPELINE_ONLY_ON_EMB_MISSING" == "1" && "$WHITENED" == "0" && "$CBIE" == "0" ]]; then
        pivot_pair_vec="${pair_embed_dir}/${PREFIX}${pair}.${data}.${PIVOT_TAG}.vec"
        trg_pair_vec="${pair_embed_dir}/${PREFIX}${pair}.${data}.${trg_lang}.vec"

        if [[ -s "$pivot_pair_vec" && -s "$trg_pair_vec" ]]; then
          echo "[skip pair][raw] $pair embeddings already exist → skipping csls/pred/eval"
          continue
        fi
      fi

      ############################################################
      # CBIE/WHITENED: skip pair ONLY if mining outputs already exist
      ############################################################
      if [[ "$PIPELINE_ONLY_ON_EMB_MISSING" == "1" && ( "$CBIE" == "1" || "$WHITENED" == "1" ) ]]; then
        if [[ -s "$sim_out" && -s "$pred" && -s "$res_file" ]]; then
          echo "[skip pair][cbie/whitened] $pair mining already exists → skipping"
          continue
        fi
      fi

      ############################################################
      # 2a) EMBEDDINGS per lang
      ############################################################
      if [[ "$WHITENED" == "0" && "$CBIE" == "0" ]]; then
        # RAW MODE: embed or reuse
        for lang in $src_lang $trg_lang; do
          out_vec="${pair_embed_dir}/${PREFIX}${pair}.${data}.${lang}.vec"

          if [[ "$lang" == "$PIVOT_TAG" ]]; then
            if [[ -s "$out_vec" ]]; then
              echo "[skip emb] $out_vec already exists (copy of pivot)"
            else
              echo "[pivot copy] $GLOBAL_PIVOT_VEC → $out_vec"
              cp "$GLOBAL_PIVOT_VEC" "$out_vec"
            fi
            continue
          fi

          if [[ -s "$out_vec" ]]; then
            echo "[skip emb] $out_vec already exists"
            continue
          fi

          if [[ "$USE_CUSTOM_DATA" == "1" ]]; then
            base_in="$SHUF_TXT_DIR/$lang.txt"
          else
            base_in="$DATA/bucc_style_data/bucc_style_${src_lang}-${trg_lang}/${src_lang}-${trg_lang}.$data.$lang"
          fi

          if [[ ! -f "$base_in" ]]; then
            echo "[warn] missing txt for $lang → $base_in (skipping embedding)"
            continue
          fi

          tmp_pair_dir="$TMP_IN/$pair"
          mkdir -p "$tmp_pair_dir"
          tsv_in="$tmp_pair_dir/$data.$lang.tsv"

          ensure_tsv "$base_in" "$tsv_in"

          echo "[emb] $pair ($lang, set=$data, backend=${BACKEND})"
          $PYTHON contextual_sentence_embeddings.py \
            --input_file "$tsv_in" \
            --output_file "$out_vec" \
            -m "${BACKEND}"
        done
      else
        # CBIE or WHITENED MODE: do NOT embed, just check they exist
        for lang in $src_lang $trg_lang; do
          out_vec="${pair_embed_dir}/${PREFIX}${pair}.${data}.${lang}.vec"
          if [[ ! -s "$out_vec" ]]; then
            if [[ "$WHITENED" == "1" ]]; then
              echo "[warn][whitened] expected vec missing: $out_vec"
            else
              echo "[warn][cbie] expected vec missing: $out_vec"
            fi
          fi
        done
      fi

      ############################################################
      # 2b) CSLS
      ############################################################
      if [[ -s "$sim_out" ]]; then
        echo "[skip csls] $sim_out already exists"
      else
        pivot_vec="${pair_embed_dir}/${PREFIX}${pair}.${data}.${PIVOT_TAG}.vec"
        other_vec="${pair_embed_dir}/${PREFIX}${pair}.${data}.${trg_lang}.vec"

        if [[ ! -s "$pivot_vec" || ! -s "$other_vec" ]]; then
          echo "[warn] missing embeddings for $pair, skipping csls"
        else
          echo "[csls] $pair (set=$data) — source=$PIVOT_TAG, target=$trg_lang"
          CUDA_VISIBLE_DEVICES=${GPUS:-0} $PYTHON scripts/bilingual_nearest_neighbor.py \
            --source_embeddings "$pivot_vec" \
            --target_embeddings "$other_vec" \
            --output "$sim_out" \
            --knn 10 -m csls --cslsknn 20
        fi
      fi

      ############################################################
      # 2c) PRED
      ############################################################
      if [[ -s "$pred" ]]; then
        echo "[skip pred] $pred already exists"
      else
        if [[ -s "$sim_out" ]]; then
          echo "[pred] using CSLS output as pred (no extra filtering)"
          cp "$sim_out" "$pred"
        else
          echo "[warn] no sim_out → no pred for $pair"
        fi
      fi

      ############################################################
      # 2d) EVAL (only if gold exists AND res missing)
      ############################################################
      gold="$PROJECT_ROOT/data/goldpairs/$PIVOT_TAG/$pair.custom.gold"

      if [[ -f "$gold" && -s "$pred" ]]; then
        if [[ -s "$res_file" ]]; then
          echo "[skip eval] $res_file already exists"
          echo -n "F1: "
          tail -n1 "$res_file" || true
        else
          echo "[eval] $pred vs $gold"
          $PYTHON scripts/bucc_f-score.py -p "$pred" -g "$gold" > "$res_file"
          echo -n "F1: "
          tail -n1 "$res_file" || true
        fi
      else
        if [[ ! -f "$gold" ]]; then
          echo "[warn] gold not found, skipped eval: $gold"
        fi
      fi

    done
  done
done
