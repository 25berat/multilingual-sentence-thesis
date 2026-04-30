import random
from pathlib import Path

# CONFIG
PROJECT_ROOT = Path(r"C:\Users\berat\PycharmProjects\PaSeMiLL")
UNSHUF_DIR = PROJECT_ROOT / "data" / "unshuffled_custom_data"
SHUF_DIR = PROJECT_ROOT / "data" / "shuffled_custom_data"

# deterministic
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

SHUF_DIR.mkdir(parents=True, exist_ok=True)


def shuffle_file(src_path: Path, dst_path: Path, perm_path: Path):
    lines = src_path.read_text(encoding="utf-8").splitlines(keepends=True)

    idxs = list(range(len(lines)))
    random.shuffle(idxs)  # independent shuffle

    # write shuffled text
    with dst_path.open("w", encoding="utf-8", newline="") as f_out:
        for new_pos in range(len(lines)):
            old_pos = idxs[new_pos]
            f_out.write(lines[old_pos])

    # write permutation: new_index \t old_index
    with perm_path.open("w", encoding="utf-8", newline="") as f_perm:
        for new_pos, old_pos in enumerate(idxs):
            f_perm.write(f"{new_pos}\t{old_pos}\n")


def main():
    if not UNSHUF_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {UNSHUF_DIR}")

    for file in sorted(UNSHUF_DIR.glob("*.txt")):
        dst_path = SHUF_DIR / file.name
        perm_path = SHUF_DIR / f"{file.stem}.perm"

        shuffle_file(file, dst_path, perm_path)
        print(f"[shuffled] {file.name} → {dst_path.name}, perm → {perm_path.name}")

    print("[done] all languages shuffled independently.")


if __name__ == "__main__":
    main()
