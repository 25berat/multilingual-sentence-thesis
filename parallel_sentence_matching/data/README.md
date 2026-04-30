# Data

This repository contains only a **small subset of the data** required to run the full pipeline.  
The complete datasets are **not included due to size constraints** and are provided separately ([Link](https://syncandshare.lrz.de/getlink/fiQhwKNzpRQctxoNkg1Qn7/data)).

---

## Folder Structure

### `unshuffled_custom_data/`
Contains the **raw, unshuffled data**.  
This is primarily used for the **source language** during experiments.

---

### `shuffled_custom_data/`
Contains the **shuffled versions of the data**, which are generated from the raw data using:

- `code/shuffle_custom_data.py`  
  → Creates shuffled datasets from the unshuffled data

All **training languages** (except the source language) are taken from this folder.

---

### `goldpairs/`
Contains gold standard alignment files for evaluation.  
These are generated using:

- `code/make_custom_gold_from_perm.py`  
  → Generates gold alignment pairs based on permutations

Currently includes subfolders for:
- `deu_Latn`
- `tur_Latn`

---

### `train/`
Training data is **not fully included** in this repository.

To reconstruct the training data:
- Use:
  - `code/gettraindata.py`
  - `code/merger.py`

Due to size constraints, the full raw datasets are not provided in the external link.  
Instead, only the **merged dataset (directly usable for training)** is included.

The original raw training data can be reconstructed using:
`code/gettraindata.py`

---

## Data Selection Logic

The data used in experiments is **automatically selected** based on the language setting (`deu_Latn` or `tur_Latn`) in:

`code/environment_full_xlmr.sh`

## Feature Encoding

For the regression analysis, linguistic feature tables are encoded using:

`encoder.py`
`encoder_one_hot.py`

The raw feature tables (for German and Turkish) are provided in the external data link above.

In this setup:

The feature tables are stored as Excel files (.xlsx)
The file paths are hardcoded inside the encoder scripts
These paths must be adapted manually if the files are stored elsewhere
