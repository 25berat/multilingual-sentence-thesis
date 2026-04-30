from __future__ import annotations

import os
from datasets import load_dataset, get_dataset_config_names


OUT_DIR = r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\train"
DATASET_ID = "cis-lmu/Glot500"

LANGS = [


]
MAX_LINES = 300_000  # cap per language


def guess_text_col(cols: list[str]) -> str:
    for c in ("text", "sentence", "sent", "content"):
        if c in cols:
            return c
    return cols[0]


def pick_split(ds) -> str:
    return "train" if "train" in ds else list(ds.keys())[0]


def export_one_language(lang_cfg: str, out_dir: str, max_lines: int | None = None) -> tuple[bool, str]:
    """
    Returns (ok, message). If ok=True, writes <lang_cfg>.txt into out_dir.
    Uses streaming=True to avoid huge downloads / disk cache issues.
    """
    try:
        ds = load_dataset(DATASET_ID, lang_cfg, streaming=True)
    except Exception as e:
        return False, f"{lang_cfg}: not available (load failed: {type(e).__name__}: {e})"

    split = pick_split(ds)
    it = ds[split]

    # Peek one element to detect columns (streaming iterable)
    try:
        first = next(iter(it))
    except StopIteration:
        return False, f"{lang_cfg}: empty split '{split}'"

    cols = list(first.keys())
    text_col = guess_text_col(cols)

    # Re-create iterator after peeking
    it = load_dataset(DATASET_ID, lang_cfg, streaming=True)[split]

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lang_cfg}.txt")

    n = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for ex in it:
            t = ex.get(text_col, None)
            if isinstance(t, str):
                t = t.strip()
                if t:
                    f.write(t.replace("\n", " ") + "\n")
                    n += 1
                    if max_lines is not None and n >= max_lines:
                        break

    return True, f"{lang_cfg}: wrote {n} lines -> {out_path} (split={split}, text_col={text_col}, streaming=True)"


def main():
    print(f"Fetching available config names for {DATASET_ID} ...")
    available = set(get_dataset_config_names(DATASET_ID))
    print(f"Available configs: {len(available)}")

    ok_count = 0
    skip_count = 0

    for lang in LANGS:
        if lang not in available:
            print(f"{lang}: not available (skipping)")
            skip_count += 1
            continue

        ok, msg = export_one_language(lang, OUT_DIR, max_lines=MAX_LINES)
        print(msg)
        if ok:
            ok_count += 1
        else:
            skip_count += 1

    print("\nDone.")
    print("Exported:", ok_count)
    print("Skipped:", skip_count)
    print("Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()