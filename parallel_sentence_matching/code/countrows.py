from __future__ import annotations

from pathlib import Path
import csv

TRAIN_DIR = Path(r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\train")
OUT_CSV = TRAIN_DIR / "language_row_counts.csv"

# Files to ignore
IGNORE = {"merged_shuffled.txt", "temp_all.txt"}
IGNORE_PREFIXES = ("_",)  # e.g., "_buckets" folder or "_something.txt"

def count_lines_fast(path: Path) -> int:
    # Fast line counting (binary read, counts b'\n')
    n = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            n += chunk.count(b"\n")
    return n

def approx_tokens_whitespace(path: Path, max_bytes: int | None = None) -> int:
    """
    Rough token estimate: counts whitespace-separated "words".
    If max_bytes is set, only reads that many bytes (faster).
    """
    tokens = 0
    read_bytes = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            if max_bytes is not None:
                if read_bytes + len(chunk) > max_bytes:
                    chunk = chunk[: max_bytes - read_bytes]
                read_bytes += len(chunk)
            # decode loosely to split on whitespace
            text = chunk.decode("utf-8", errors="ignore")
            tokens += len(text.split())
            if max_bytes is not None and read_bytes >= max_bytes:
                break
    return tokens

def main():
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {TRAIN_DIR}")

    rows = []
    for p in sorted(TRAIN_DIR.glob("*.txt")):
        if p.name in IGNORE:
            continue
        if p.name.startswith(IGNORE_PREFIXES):
            continue

        lang = p.stem  # e.g., "deu_Latn"
        line_count = count_lines_fast(p)
        size_mb = p.stat().st_size / (1024 * 1024)

        rows.append({
            "language": lang,
            "filename": p.name,
            "lines": line_count,
            "size_mb": round(size_mb, 2),
            # optional: token estimate (comment out if you don't need it)
            # "approx_tokens_ws": approx_tokens_whitespace(p),
        })
        print(f"{lang}: {line_count:,} lines  ({size_mb:.2f} MB)")

    # Write CSV
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["language","filename","lines","size_mb"])
        w.writeheader()
        w.writerows(rows)

    print("\nDone.")
    print(f"Wrote: {OUT_CSV}")
    print(f"Total languages/files: {len(rows)}")

if __name__ == "__main__":
    main()