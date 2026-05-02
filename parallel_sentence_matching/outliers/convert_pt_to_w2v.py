import os
import random
import torch

# --------- CONFIGURE THESE ---------
INPUT_PT = r"C:\Users\berat\PycharmProjects\outliers\embs\bucc\cis-lmu\glot500-base\7\als_Latn\emb.pt"
OUTPUT_W2V = r"C:\Users\berat\PycharmProjects\outliers\embs\bucc\cis-lmu\glot500-base\7\als_Latn\embw2v"
OUTPUT_W2V_SHUF = r"C:\Users\berat\PycharmProjects\outliers\embs\bucc\cis-lmu\glot500-base\7\als_Latn\embw2v.shuf"
SHUFFLE_SEED = 42
# -----------------------------------


def load_pt_embeddings(path):
    obj = torch.load(path, map_location="cpu")

    # detect structure
    if isinstance(obj, torch.Tensor):
        emb = obj
        n, d = emb.shape
        ids = [str(i) for i in range(n)]
    elif isinstance(obj, dict):
        # most common keys
        if "emb" in obj:
            emb = obj["emb"]
        elif "embeddings" in obj:
            emb = obj["embeddings"]
        else:
            raise ValueError(f"Don't know which key holds the embeddings in dict: {obj.keys()}")

        if "ids" in obj:
            ids = [str(x) for x in obj["ids"]]
        elif "sent_ids" in obj:
            ids = [str(x) for x in obj["sent_ids"]]
        else:
            n = emb.shape[0]
            ids = [str(i) for i in range(n)]
    else:
        raise ValueError(f"Unsupported object type in {path}: {type(obj)}")

    emb = emb.detach().cpu()
    n, d = emb.shape
    print(f"[convert] loaded {n} vectors of dim {d}")
    return ids, emb


def write_w2v(ids, emb, out_path):
    n, d = emb.shape
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{n} {d}\n")
        for _id, vec in zip(ids, emb):
            vec_str = " ".join(f"{x:.6f}" for x in vec.tolist())
            f.write(f"{_id} {vec_str}\n")
    print(f"[convert] wrote w2v file to {out_path}")


def shuffle_w2v(in_path, out_path, seed=42):
    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        print("[shuffle] empty file, skipping")
        return

    header = lines[0].strip()
    emb_lines = [l.strip() for l in lines[1:] if l.strip()]

    random.seed(seed)
    random.shuffle(emb_lines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in emb_lines:
            f.write(line + "\n")

    print(f"[shuffle] wrote shuffled w2v to {out_path}")


def main():
    ids, emb = load_pt_embeddings(INPUT_PT)
    write_w2v(ids, emb, OUTPUT_W2V)
    shuffle_w2v(OUTPUT_W2V, OUTPUT_W2V_SHUF, SHUFFLE_SEED)
    print("✅ done.")


if __name__ == "__main__":
    main()
