#!/usr/bin/env python3
import json
from pathlib import Path

TO_CLEAN = Path("output/localizer-data-Qwen3/out_large_gpt-4.1-mini.json")
TO_REMOVE = Path("output/localizer-data-Qwen3/sample_4.json") 
OUTPUT   = TO_CLEAN

def main():
    with TO_CLEAN.open() as f:
        data = json.load(f)
    with TO_REMOVE.open() as f:
        remove_list = json.load(f)
        
    remove_ids = {item["question_id"] for item in remove_list}

    print(f"Loaded {len(data)} items from {TO_CLEAN}")
    print(f"Loaded {len(remove_list)} items from {TO_REMOVE}")
    print(f"Will remove {len(remove_ids)} question_ids")

    cleaned = [item for item in data if item.get("question_id") not in remove_ids]

    print(f"After filtering: {len(cleaned)} items remain")

    with OUTPUT.open("w") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Saved cleaned data to {OUTPUT}")

if __name__ == "__main__":
    main()