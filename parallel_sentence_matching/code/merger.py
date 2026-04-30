import os
import random
from pathlib import Path

train_dir = Path(r"C:\Users\berat\PycharmProjects\PaSeMiLL\data\train")
output_file = train_dir / "merged_shuffled.txt"

SEED = 42
MAX_PER_LANG = 25_000

random.seed(SEED)

all_lines = []
stats = []

for filename in os.listdir(train_dir):
    if not filename.endswith(".txt"):
        continue
    if filename in {output_file.name, "temp_all.txt"}:
        continue
    if filename.startswith("_"):
        continue

    file_path = train_dir / filename
    print("Reading:", file_path.name)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    total_lines = len(lines)

    if total_lines <= MAX_PER_LANG:
        selected_lines = lines
    else:
        selected_lines = random.sample(lines, MAX_PER_LANG)

    all_lines.extend(selected_lines)
    stats.append((filename, total_lines, len(selected_lines)))

    print(f"  total={total_lines:,} -> selected={len(selected_lines):,}")

print(f"\nTotal selected lines before shuffle: {len(all_lines):,}")

print("Shuffling all lines...")
random.shuffle(all_lines)

with open(output_file, "w", encoding="utf-8") as out:
    for line in all_lines:
        out.write(line + "\n")

print(f"\nDone: {output_file}")
print(f"Final total lines: {len(all_lines):,}")

print("\nPer-language stats:")
for fname, total, selected in sorted(stats):
    print(f"{fname}: total={total:,}, selected={selected:,}")