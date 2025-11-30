#!/usr/bin/env python3
import json
from pathlib import Path

TRAIN_PATH = Path("output/Qwen3-TrainTest-data/train-pre-with-plan.json")
# TEST_PATH  = Path("output/Qwen3-TrainTest-data/test-pre.json")
TEST_PATH=Path("none.json")

OUT_TRAIN = Path("output/Qwen3-TrainTest-data/modular-step-by-step/train-planner.json")
# OUT_TEST  = Path("output/Qwen3-TrainTest-data/test-final.json")
OUT_TEST=Path("none.json")

TEST_WHITELIST = {
    "question_title",
    "question_content",
    "code_list",
}

#localizer
# TRAIN_WHITELIST = {
#     "question_title",
#     "question_content",
#     "metadata",
#     "code_list",
#     "bug_span",
#     "bug_summary"
# }

#planner
TRAIN_WHITELIST = {
    "question_title",
    "question_content",
    "metadata",
    "code_list",
    "bug_span",
    "bug_summary",
    "planner_text"
}

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def save_jsonl(path, data):
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def clean_train_item(item):
    """Keep only allowed fields."""
    return {k: item[k] for k in TRAIN_WHITELIST if k in item}

def clean_test_item(item):
    """Keep only allowed fields."""
    return {k: item[k] for k in TEST_WHITELIST if k in item}

def main():
    if TRAIN_PATH.exists():
        with TRAIN_PATH.open() as f:
            train_data = json.load(f)
        print(f"[TRAIN] Loaded {len(train_data)} samples.")
        
        cleaned_train = []
        for item in train_data:
            item = clean_train_item(item)
            cleaned_train.append(item)
        with OUT_TRAIN.open("w") as f:
            json.dump(cleaned_train, f, indent=2, ensure_ascii=False)
        print(f"[TRAIN] Saved cleaned train → {OUT_TRAIN}")

    if TEST_PATH.exists():
        with TEST_PATH.open() as f:
            test_data = json.load(f)
        print(f"[TEST] Loaded {len(test_data)} samples.")

        cleaned_test = []
        for item in test_data:
            item = clean_test_item(item)
            cleaned_test.append(item)

        with OUT_TEST.open("w") as f:
            json.dump(cleaned_test, f, indent=2, ensure_ascii=False)
        print(f"[TEST] Saved cleaned test → {OUT_TEST}")

    print("Done!")


if __name__ == "__main__":
    main()
