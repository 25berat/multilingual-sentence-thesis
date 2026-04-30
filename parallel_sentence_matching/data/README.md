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
Contains the **shuffled versions of the data**.  
All **training languages** (except the source language) are taken from this folder.

---

### `goldpairs/`
Contains gold standard alignment files for evaluation.  
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

The data used in experiments is **automatically selected** based on the language setting in:

```bash
code/environment_full_xlmr.sh


---

## Feature Encoding

The linguistic feature tables used for the regression analysis are processed with:

- `encoder.py`
- `encoder_one_hot.py`

In this setup, the raw feature tables were provided as Excel files (`.xlsx`).  
The paths to these raw feature tables are currently **hardcoded inside the encoder scripts** and must be changed manually if the files are stored in a different location.

The encoder scripts generate encoded feature tables that can then be used for the later statistical analysis.
