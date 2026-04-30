#!/usr/bin/env python3
import argparse
import logging
import re
# top of make_custom_gold.py
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("RESULTS", PROJECT_ROOT / "results"))
TMP_DIR = RESULTS_DIR / "tmp_inputs"
GOLD_DIR = PROJECT_ROOT / "data" / "goldpairs"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

WS_SPLIT = re.compile(r'\s+')
SRC_PREFIX = re.compile(r'^(src-)', re.IGNORECASE)
TRG_PREFIX = re.compile(r'^(trg-)', re.IGNORECASE)
LEADING_ZEROS = re.compile(r'^0+')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--prediction', required=True,
                    help='Pred: srcID trgID [score]; tabs or spaces; with/without src-/trg- prefixes.')
    ap.add_argument('-g', '--gold', required=True,
                    help='Gold: srcID trgID; tabs or spaces; with/without src-/trg- prefixes.')
    return ap.parse_args()

def canonicalize_id(x: str, is_src: bool) -> str:
    """Strip src-/trg- prefix, remove leading zeros, return canonical numeric-ish string.
       If non-numeric leftovers remain, we keep them as-is (still consistent if both files match)."""
    x = x.strip()
    x = SRC_PREFIX.sub('', x) if is_src else TRG_PREFIX.sub('', x)
    # If it's numeric-like, strip leading zeros; keep at least '0' if all zeros
    if x.isdigit():
        x = LEADING_ZEROS.sub('', x) or '0'
    return x

def iter_pairs(path: str, is_gold: bool):
    """Yield (src, trg) from a file.
       - Accept tabs/spaces, optional score col
       - Skip empty lines, comments (#...), and CSV-like headers
       - Swallow BOM via utf-8-sig
    """
    with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
        for ln, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # skip obvious CSV header lines like 'PRECISION,RECALL,F1'
            if ',' in line and not WS_SPLIT.search(line):
                continue

            parts = WS_SPLIT.split(line)
            if len(parts) < 2:
                logger.warning("Skipping malformed line %d in %s: <%s>", ln, path, line)
                continue

            src = canonicalize_id(parts[0], is_src=True)
            trg = canonicalize_id(parts[1], is_src=False)
            yield src, trg

def load_gold(path: str):
    gold = {}
    for s, t in iter_pairs(path, is_gold=True):
        if s in gold and gold[s] != t:
            logger.warning("Gold duplicate src with different trg; keeping first: %s (old=%s, new=%s)", s, gold[s], t)
            continue
        gold[s] = t
    return gold

def load_pred(path: str):
    pred = {}
    for s, t in iter_pairs(path, is_gold=False):
        if s in pred and pred[s] != t:
            logger.warning("Pred duplicate for src %s; keeping first=%s, ignoring new=%s", s, pred[s], t)
            continue
        pred[s] = t
    return pred

def main(prediction: str, gold: str):
    gold_map = load_gold(gold)
    pred_map = load_pred(prediction)

    N = len(gold_map)
    tp = 0
    fp = 0

    for s, t_pred in pred_map.items():
        t_gold = gold_map.get(s)
        if t_gold is None:
            fp += 1
        elif t_pred == t_gold:
            tp += 1
        else:
            fp += 1

    fn = N - tp

    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = (2 * P * R) / (P + R) if (P + R) else 0.0

    print('PRECISION,RECALL,F1')
    print(f'{P},{R},{F1}')

if __name__ == '__main__':
    args = parse_args()
    main(args.prediction, args.gold)
