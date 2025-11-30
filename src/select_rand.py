#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Randomly split JSON list into 4 non-overlapping samples of 25 each.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input JSON file (a list of objects).")
    ap.add_argument("--out-prefix", default="sample", help="Output file prefix (default: sample).")
    ap.add_argument("--groups", type=int, default=4, help="Number of groups to create (default: 4).")
    ap.add_argument("--size", type=int, default=25, help="Size of each group (default: 25).")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility (optional).")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    need = args.groups * args.size
    if len(data) < need:
        raise ValueError(f"Not enough items: need {need}, but input has {len(data)}.")

    if args.seed is not None:
        random.seed(args.seed)

    # Shuffle a copy and take the first `need` items, then chunk.
    shuffled = data[:]  # shallow copy
    random.shuffle(shuffled)
    selected = shuffled[:need]

    # Write each chunk to its own file.
    for i in range(args.groups):
        chunk = selected[i*args.size:(i+1)*args.size]
        out_path = Path(f"{args.out_prefix}_{i+1}.json")
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(chunk)} items to {out_path}")

if __name__ == "__main__":
    main()
