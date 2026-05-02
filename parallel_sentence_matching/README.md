# BUCC Mining Pipeline

This pipeline runs the full BUCC-style mining workflow automatically once the input data and paths are set up correctly.
If the input is correct, the whole pipeline runs on its own, from computing embeddings to nearest neighbor retrieval and evaluation.

## Overview

The pipeline supports:

- RAW mode: compute embeddings and run mining
- CBIE mode: use CBIE-transformed vectors
- WHITENED mode: use whitened vectors

Important: CBIE and WHITENED can only be used after RAW has been run first, because the raw vectors must already exist before they can be postprocessed.

## Main scripts

### 1. environment_full_xlmr.sh

This script defines the main environment variables, default paths, result folders, and backend settings.

Important variables:

PIVOT_TAG="${PIVOT_TAG:-deu_Latn}"   # or tur_Latn
BACKEND="${BACKEND:-pretrained}"

Supported backend / model names:

- glot500
- labse
- xlmr
- laser
- sonar
- qwen3
- llama31
- mmbert
- pretrained   (trained mMBERT)
- studentteacher

### 2. mine_bucc_full_xlmr.sh

This is the main pipeline script.

Important variables:

PIVOT_TAG=${PIVOT_TAG:-"deu_Latn"}   # single pivot variable
CBIE=${CBIE:-0}         # 0 = RAW run (compute embeddings); 1 = use CBIE vecs only
WHITENED=${WHITENED:-0} # 1 = use WHITENED vecs only

If the input files are set up correctly, this script automatically runs the full pipeline:

1. load the pivot and target-language text files
2. convert them into TSV input format
3. compute sentence embeddings in RAW mode
4. run nearest neighbor mining with CSLS
5. create prediction files
6. run evaluation if gold files are available

So if the input is correct, the whole pipeline runs automatically from embeddings to mining and evaluation.

### 3. CBIE mode / Whitening

CBIE and WHITENED mode do not compute embeddings from scratch.

Instead, they expect CBIE-transformed or whitened vectors to already exist.

Before using CBIE or WHITENED mode, you must:

1. run the pipeline in RAW mode first
2. create CBIE vectors using:

python apply_cbie_to_existing_vecs.py

3. create whitened vectors using:

python export_whitened_vecs.py

Then you can run the mining pipeline in CBIE or WHITENED mode.

Important:
RAW must always be run first.

### 4. TTR

For type-token ratio (TTR) computation, use:

python compute_ttr_whitespace.py

This script computes TTR based on whitespace tokenization.

## Data setup

There is also a separate README for the data setup in the data folder.

Please read that first before running the mining pipeline.

The mining pipeline assumes that the data already exists in the expected folder structure and naming format.

## Path adjustments

The paths in the scripts may have to be adjusted depending on your local setup or cluster setup.

In particular, check and adapt if needed:

- project root paths
- data paths
- results paths
- temporary input/output paths
- third-party tool paths
- gold file paths

Before running the pipeline, review:

- environment_full_xlmr.sh
- mine_bucc_full_xlmr.sh

and adapt the paths if necessary.
