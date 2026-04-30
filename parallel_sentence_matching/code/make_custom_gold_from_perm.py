from pathlib import Path

# =========================
# CONFIG (ONLY CHANGE THIS)
# =========================
PIVOT_TAG = "deu_Latn"   # e.g. "deu_Latn", "tur_Latn", ...

PROJECT_ROOT = Path(r"C:\Users\berat\PycharmProjects\PaSeMiLL")
UNSHUF_DIR = PROJECT_ROOT / "data" / "unshuffled_custom_data"
SHUF_DIR   = PROJECT_ROOT / "data" / "shuffled_custom_data"
GOLD_ROOT  = PROJECT_ROOT / "data" / "goldpairs" / PIVOT_TAG

# optional: skip these languages (stems)
EXCLUDE = set()  # e.g. {"ceb_Latn"}

GOLD_ROOT.mkdir(parents=True, exist_ok=True)


def load_perm_new_to_old(perm_path: Path) -> dict[int, int]:
    """
    .perm: each line = new_index<TAB>old_index
    return dict: new -> old
    """
    d: dict[int, int] = {}
    with perm_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            new_str, old_str = line.split("\t")
            d[int(new_str)] = int(old_str)
    return d


def main():
    if not UNSHUF_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {UNSHUF_DIR}")
    if not SHUF_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {SHUF_DIR}")

    pivot_file = UNSHUF_DIR / f"{PIVOT_TAG}.txt"
    if not pivot_file.exists():
        raise FileNotFoundError(f"Pivot (unshuffled) file not found: {pivot_file}")

    pivot_len = sum(1 for _ in pivot_file.open("r", encoding="utf-8"))
    print(f"[pivot] {PIVOT_TAG} is UNshuffled (len={pivot_len})")
    print(f"[out]   {GOLD_ROOT}")

    # iterate over shuffled languages
    for shuf_txt in sorted(SHUF_DIR.glob("*.txt")):
        lang = shuf_txt.stem

        # IMPORTANT: skip pivot language in shuffled dir
        if lang == PIVOT_TAG:
            print(f"[skip] {lang} (pivot should be unshuffled; ignoring shuffled copy)")
            continue
        if lang in EXCLUDE:
            continue

        perm_path = SHUF_DIR / f"{lang}.perm"
        if not perm_path.exists():
            print(f"[warn] missing perm for {lang}, skipping")
            continue

        new_to_old = load_perm_new_to_old(perm_path)

        # We need: pivot_old_idx \t lang_new_idx
        # Because pivot is unshuffled, pivot index == old index
        pairs = []
        for lang_new, lang_old in new_to_old.items():
            if lang_old >= pivot_len:
                continue
            pairs.append((lang_old, lang_new))

        # Sort by pivot_old_idx for nice monotonic gold
        pairs.sort(key=lambda x: x[0])

        gold_path = GOLD_ROOT / f"{PIVOT_TAG}-{lang}.custom.gold"
        with gold_path.open("w", encoding="utf-8", newline="") as g:
            for pivot_old, lang_new in pairs:
                g.write(f"{pivot_old}\t{lang_new}\n")

        print(f"[gold] {gold_path.name} (pairs={len(pairs)})")

    print("[done] gold generation finished.")


if __name__ == "__main__":
    main()
