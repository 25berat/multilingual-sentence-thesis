#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import torch

# >>> Adjust this to your root directory:
BASE = Path(r"C:\Users\berat\PycharmProjects\outliers\embs\bucc\cis-lmu\glot500-base\7")

DRY_RUN = False  # set to True to preview, False to perform changes


# ----------------- basic fs helpers -----------------
def log(msg):
    print(msg)


def rm(path: Path):
    if path.exists():
        log(f"[DEL] {path}")
        if not DRY_RUN:
            path.unlink()


def mv(src: Path, dst: Path):
    if src.resolve() == dst.resolve():
        return
    log(f"[REN] {src}  ->  {dst}")
    if not DRY_RUN:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dst)


def rename_dir(src: Path, dst: Path):
    if src.resolve() == dst.resolve():
        return
    log(f"[DIR] {src.name}  ->  {dst.name}")
    if not DRY_RUN:
        if dst.exists():
            # move contents into existing dst, then remove src
            for p in src.iterdir():
                mv(p, dst / p.name)
            src.rmdir()
        else:
            src.rename(dst)


# ----------------- STEP 1: fix pair dirs -----------------
def process_pair_dir(pair_dir: Path):
    """
    Handles two cases:

    1) SPECIAL CASE: deu_Latn
       - expects a file: *.deu_Latn.custom.deu_Latn.vec
       - renames that file to emb.pt
       - renames folder deu_Latn -> deu_Latn

    2) Normal pair dir: 'deu_Latn-<target>'
       - deletes *.custom.deu_Latn.vec
       - keeps *.custom.<target>.vec as emb.pt
       - renames folder 'deu_Latn-<target>' -> '<target>'
    """

    # >>> SPECIAL RULE FOR deu_Latn <<<
    if pair_dir.name == "deu_Latn":
        # look for something like: {model}.deu_Latn.custom.deu_Latn.vec
        candidates = list(pair_dir.glob("*.deu_Latn.custom.deu_Latn.vec"))
        if not candidates:
            log(f"[WARN] deu_Latn: no '*.deu_Latn.custom.deu_Latn.vec' found in {pair_dir}")
        else:
            src = candidates[0]
            dst_file = pair_dir / "emb.pt"

            # rename that file to emb.pt
            mv(src, dst_file)

            # rename folder to deu_Latn
            new_dir = pair_dir.parent / "deu_Latn"
            rename_dir(pair_dir, new_dir)

        # deu_Latn handled completely, don't run normal logic
        return

    # >>> NORMAL LOGIC FOR deu_Latn-<target> <<<
    if not pair_dir.is_dir():
        return
    name = pair_dir.name
    if "-" not in name:
        return
    left, target = name.split("-", 1)
    if left != "deu_Latn":
        return

    # delete the "second" files (*.custom.deu_Latn.vec)
    for f in pair_dir.glob("*.custom.deu_Latn.vec"):
        rm(f)

    # rename kept file (*.custom.<target>.vec -> emb.pt)
    kept_candidates = sorted(pair_dir.glob(f"*.custom.{target}.vec"))
    if not kept_candidates:
        log(f"[WARN] No '*.custom.{target}.vec' in {pair_dir}")
    else:
        mv(kept_candidates[0], pair_dir / "emb.pt")
        # if multiple candidates, keep only first and delete the rest
        for extra in kept_candidates[1:]:
            rm(extra)

    # rename folder to just <target>
    rename_dir(pair_dir, pair_dir.parent / target)


def step1_fix_dirs(base: Path):
    # process normal deu_Latn-* dirs
    pair_dirs = [d for d in base.iterdir() if d.is_dir() and d.name.startswith("deu_Latn-")]
    # also explicitly check for deu_Latn if it exists
    global_deu = base / "deu_Latn"
    if global_deu.is_dir():
        pair_dirs.append(global_deu)

    if not pair_dirs:
        log(f"[INFO] No 'deu_Latn-*' or 'deu_Latn' folders under {base}")
        return

    for d in sorted(pair_dirs):
        process_pair_dir(d)


# ----------------- STEP 2: convert text emb.pt -> torch -----------------
def looks_like_torch_binary(p: Path) -> bool:
    """True if torch.load succeeds."""
    try:
        _ = torch.load(p, map_location="cpu")
        return True
    except Exception:
        return False


def load_text_embeddings(path: Path) -> np.ndarray:
    """
    Supports:
      - optional first line header 'N D'
      - rows: 'id f1 f2 ... fD'  (drops the id column if it looks integer-like)
    """
    with path.open("r", encoding="utf-8") as f:
        first = f.readline().strip().split()
    # check if first line is a header like "N D"
    skip = 1 if len(first) == 2 and all(tok.isdigit() for tok in first) else 0

    arr = np.loadtxt(path, dtype=float, ndmin=2, skiprows=skip)

    # drop ID column if first column is integer-like
    if arr.ndim == 2 and arr.shape[1] > 2:
        c0 = arr[: min(10, len(arr)), 0]
        if np.allclose(c0, np.round(c0)):
            arr = arr[:, 1:]

    return arr


def convert_emb_file(p: Path):
    log(f"[CHK] {p}")
    if looks_like_torch_binary(p):
        log("  -> already torch binary; skip.")
        return

    # try parse as text
    try:
        arr = load_text_embeddings(p)
    except Exception as e:
        log(f"  [ERR] Could not parse as text: {e}")
        return

    log(f"  -> convert text -> torch ({arr.shape[0]} x {arr.shape[1]})")
    if not DRY_RUN:
        tens = torch.from_numpy(arr.astype(np.float32))
        torch.save({"emb": tens}, p)


def step2_convert_all(base: Path):
    for f in sorted(base.rglob("emb.pt")):
        convert_emb_file(f)


# ----------------- main -----------------
def main():
    if not BASE.exists():
        log(f"[ERR] Base path does not exist: {BASE}")
        sys.exit(1)

    log(f"[INFO] DRY_RUN={DRY_RUN}  |  Base={BASE}")

    # 1) fix directory/file layout
    step1_fix_dirs(BASE)

    # 2) convert any text emb.pt to torch binary
    step2_convert_all(BASE)

    log("[DONE] Preview complete." if DRY_RUN else "[DONE] Changes applied.")


if __name__ == "__main__":
    main()
