# reverse_rename_gold_files.py
# Renames every ...\bucc_ready\<lang>\deu_Latn-<lang>.custom.gold  ->  gold.txt
# Only the filename changes; it stays in the same folder.

from __future__ import annotations
from pathlib import Path


def main() -> None:
    root = Path(r"C:\Users\berat\PycharmProjects\Belopsem\raw_data\bucc_ready")
    pivot = "deu_Latn"

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    # matches: deu_Latn-<lang>.custom.gold (lang is the folder name)
    candidates = list(root.rglob(f"{pivot}-*.custom.gold"))

    if not candidates:
        print(f"No '{pivot}-*.custom.gold' files found under: {root}")
        return

    renamed = 0
    skipped = 0

    for p in candidates:
        lang_tag = p.parent.name  # folder name, e.g. ace_Arab
        expected_name = f"{pivot}-{lang_tag}.custom.gold"

        # Safety: only touch files that match the folder name
        if p.name != expected_name:
            print(f"[SKIP] Name doesn't match folder: {p} (expected {expected_name})")
            skipped += 1
            continue

        target = p.with_name("gold.txt")

        if target.exists():
            print(f"[SKIP] gold.txt already exists in folder: {target}")
            skipped += 1
            continue

        p.rename(target)
        print(f"[OK] {p.name}  ->  {target.name}  (in {lang_tag})")
        renamed += 1

    print(f"\nDone. Renamed: {renamed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
