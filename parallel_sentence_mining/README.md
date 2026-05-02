This pipeline runs the full parallel sentence mining workflow automatically once the input data and paths are set up correctly.
If the input is correct, the whole pipeline runs on its own, from computing embeddings to nearest neighbor retrieval and evaluation.

## Overview

This pipeline currently supports only:

- RAW mode: compute embeddings and run mining

There are no CBIE or WHITENED modes here.
The pipeline uses the raw embeddings directly.

## Main scripts

### 1. environment_full_xlmr.sh

This script defines the main environment variables, default paths, result folders, and backend settings.

Important variables:

PIVOT_TAG="${PIVOT_TAG:-deu_Latn}"   # or tur_Latn
BACKEND="${BACKEND:-pretrained}"

Supported backend / model names:

- xlmr
- glot500
- labse
- sonar
- laser
- qwen3
- llama31

### 2. mine_bucc_full_xlmr.sh

This is the main pipeline script.

Important variable:

PIVOT_TAG=${PIVOT_TAG:-"deu_Latn"}   # single pivot variable

If the input files are set up correctly, this script automatically runs the full pipeline:

1. load the pivot and target-language text files
2. convert them into TSV input format
3. compute sentence embeddings
4. run nearest neighbor mining with CSLS
5. create prediction files
6. run evaluation if gold files are available

So if the input is correct, the whole pipeline runs automatically from embeddings to mining and evaluation.

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
