# Data

This repository contains only a **small subset of the data** required to run the full pipeline.  
The complete datasets are **not included due to size constraints** and will be provided separately (link will be added later).

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
- These scripts generate the required training files, including the **merged large dataset**

---

## Data Selection Logic

The data used in experiments is **automatically selected** based on the language setting (`deu_Latn` or `tur_Latn`) in:

```bash
code/environment_full_xlmr.sh
