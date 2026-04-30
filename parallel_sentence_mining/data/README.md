# Raw Data (Mining)

This folder contains the input data used for the **parallel sentence mining pipeline**.

Due to size constraints, the full dataset is not included in this repository.  
The complete data can be downloaded here:

👉 https://syncandshare.lrz.de/getlink/fiX5XJ3wZ11QWorB3zN6Dr/raw_data

---

## Folder Structure

### `bucc_ready/`
Contains data that has been **prepared in BUCC format** and can be used directly as input to the mining pipeline.

This is the **main input used by the pipeline**.

---

### `monolingual_data/` and `parallel_data/`
These folders contain the **original source data** before BUCC-style preparation.

- `monolingual_data/`: raw monolingual corpora
- `parallel_data/`: aligned or semi-aligned sentence pairs

These datasets are used to:
- construct the BUCC-style data
- generate evaluation sets (gold pairs)

