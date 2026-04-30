from pathlib import Path
import random
import re
from typing import List, Tuple, Optional

# ============================================================
# PATHS (YOUR PROJECT)
# ============================================================
BASE = Path(r"C:\Users\berat\PycharmProjects\Belopsem\raw_data")

MONO_DIR = BASE / "monolingual_data"     # flat: deu_Latn.txt, ace_Arab.txt, ...
PARA_DIR = BASE / "parallel_data"        # flat: deu_Latn.txt, ace_Arab.txt, ...
OUT_DIR  = BASE / "bucc_ready"           # output per target language

SRC_LANG = "deu_Latn"
ENC = "utf-8"
SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# REMOVE EXISTING LINE NUMBERS FROM INPUT
# Examples removed:
#   "3473 The sentence..."
#   "3473\tThe sentence..."
#   "3473 ; The sentence..."
# ============================================================
LEADING_NUM_RE = re.compile(r"^\s*\d+\s*(?:[\t ]+|[;:)\].,\-–—|/|\\|_]+\s*)")

def normalize_text(s: str) -> str:
    s = str(s).replace("\r", " ").replace("\n", " ")
    s = " ".join(s.split()).strip()
    s = LEADING_NUM_RE.sub("", s).strip()
    return s

def read_txt_lines(fp: Path) -> List[str]:
    lines = fp.read_text(encoding=ENC, errors="replace").splitlines()
    out = []
    for line in lines:
        line = normalize_text(line)
        if line:
            out.append(line)
    return out

# ============================================================
# BUCC CREATOR (NO IDs IN TEXT FILES)
# - text files contain ONLY sentences (pure text)
# - gold file contains line-number pairs for injected parallel
# ============================================================
def create_dataset_bucc_no_ids(
    mono_src: List[str],
    mono_trg: List[str],
    para_src: List[str],
    para_trg: List[str],
    seed: int = 42,
) -> Tuple[str, str, str]:
    """
    Output:
      src_text: sentence per line (no IDs!)
      trg_text: sentence per line (no IDs!)
      gold:     "src_line_index\\ttrg_line_index" for injected parallel only

    Line indices start at 1.
    """

    if len(para_src) != len(para_trg):
        raise ValueError(f"Parallel mismatch: {len(para_src)} != {len(para_trg)}")

    mono_src = [normalize_text(x) for x in mono_src if normalize_text(x)]
    mono_trg = [normalize_text(x) for x in mono_trg if normalize_text(x)]

    # Clean parallel pairwise (keep alignment)
    pairs = []
    for s, t in zip(para_src, para_trg):
        s2, t2 = normalize_text(s), normalize_text(t)
        if s2 and t2:
            pairs.append((s2, t2))

    if not pairs:
        raise RuntimeError("Parallel data is empty after cleaning!")

    # Tag injected parallel sentences with pair_id (so gold stays correct even with duplicates)
    src_items: List[Tuple[str, Optional[int]]] = [(s, None) for s in mono_src]
    trg_items: List[Tuple[str, Optional[int]]] = [(t, None) for t in mono_trg]

    for pid, (s, t) in enumerate(pairs):
        src_items.append((s, pid))
        trg_items.append((t, pid))

    # Shuffle independently
    rng = random.Random(seed)
    rng.shuffle(src_items)
    rng.shuffle(trg_items)

    # Build pure text outputs and maps: pair_id -> line_index
    src_lines: List[str] = []
    trg_lines: List[str] = []
    src_map = {}
    trg_map = {}

    for idx, (txt, pid) in enumerate(src_items, start=1):
        src_lines.append(txt)  # NO ID
        if pid is not None:
            src_map[pid] = idx

    for idx, (txt, pid) in enumerate(trg_items, start=1):
        trg_lines.append(txt)  # NO ID
        if pid is not None:
            trg_map[pid] = idx

    # Gold ONLY for injected parallel pairs
    gold_lines = [f"{src_map[i]}\t{trg_map[i]}" for i in range(len(pairs))]

    return (
        "\n".join(src_lines) + "\n",
        "\n".join(trg_lines) + "\n",
        "\n".join(gold_lines) + "\n",
    )

# ============================================================
# MAIN
# ============================================================
def main():
    print("=== BUCC creator (flat files, NO IDs in text) ===")
    print(f"MONO_DIR: {MONO_DIR}")
    print(f"PARA_DIR: {PARA_DIR}")
    print(f"OUT_DIR : {OUT_DIR}")
    print("================================================\n")

    # --- Load monolingual German ---
    mono_de_fp = MONO_DIR / f"{SRC_LANG}.txt"
    if not mono_de_fp.exists():
        raise RuntimeError(f"Missing monolingual German: {mono_de_fp}")

    mono_de = read_txt_lines(mono_de_fp)

    # --- Load parallel German (paired with ALL target parallel files) ---
    para_de_fp = PARA_DIR / f"{SRC_LANG}.txt"
    if not para_de_fp.exists():
        raise RuntimeError(
            f"Missing parallel German: {para_de_fp}\n"
            f"parallel_data MUST contain {SRC_LANG}.txt"
        )

    para_de = read_txt_lines(para_de_fp)

    # --- Iterate all target parallel languages (everything except German) ---
    para_targets = sorted([fp for fp in PARA_DIR.glob("*.txt") if fp.stem != SRC_LANG])
    if not para_targets:
        raise RuntimeError(f"No target parallel files found in: {PARA_DIR}")

    ok = 0
    for trg_fp in para_targets:
        trg_lang = trg_fp.stem

        # Monolingual target must exist
        mono_trg_fp = MONO_DIR / f"{trg_lang}.txt"
        if not mono_trg_fp.exists():
            print(f"[SKIP] Missing monolingual target: {mono_trg_fp.name}")
            continue

        mono_trg = read_txt_lines(mono_trg_fp)
        para_trg = read_txt_lines(trg_fp)

        # Parallel must be line-aligned
        if len(para_de) != len(para_trg):
            print(f"[SKIP] Parallel length mismatch for {trg_lang}: deu={len(para_de)} vs trg={len(para_trg)}")
            continue

        src_text, trg_text, gold_text = create_dataset_bucc_no_ids(
            mono_src=mono_de,
            mono_trg=mono_trg,
            para_src=para_de,
            para_trg=para_trg,
            seed=SEED,
        )

        out_lang_dir = OUT_DIR / trg_lang
        out_lang_dir.mkdir(parents=True, exist_ok=True)

        # pure text files (NO IDs)
        (out_lang_dir / f"{SRC_LANG}.txt").write_text(src_text, encoding=ENC)
        (out_lang_dir / f"{trg_lang}.txt").write_text(trg_text, encoding=ENC)

        # gold = ONLY injected parallel pairs
        (out_lang_dir / "gold.txt").write_text(gold_text, encoding=ENC)

        print(f"[OK] {trg_lang} -> wrote {out_lang_dir} | injected parallel pairs={len(para_de)}")
        ok += 1

    print(f"\nDONE. Created datasets for {ok} languages.")

if __name__ == "__main__":
    main()
