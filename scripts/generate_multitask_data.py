import json

LOCAL_PATH = "output/Qwen3-TrainTest-data/modular-step-by-step/train-localizer.json"
PLANNER_PATH = "output/Qwen3-TrainTest-data/modular-step-by-step/train-planner.json"
OUT_PATH = "output/Qwen3-TrainTest-data/multitask-step-by-step/train-multitask.json"

local_data = json.load(open(LOCAL_PATH))
planner_data = json.load(open(PLANNER_PATH))

multi = []

# ---- localizer ----
for item in local_data:
    multi.append({
        "task": "localizer",
        "question_title": item.get("question_title", ""),
        "question_content": item.get("question_content", ""),
        "code_list": item.get("code_list", []),
        "metadata": item.get("metadata", []),

        "label_bug_span": item["bug_span"],
        "label_bug_summary": item["bug_summary"]
    })

# ---- planner ----
for item in planner_data:
    multi.append({
        "task": "planner",
        "question_title": item.get("question_title", ""),
        "question_content": item.get("question_content", ""),
        "code_list": item.get("code_list", []),
        "metadata": item.get("metadata", []),

        "bug_span": item["bug_span"],
        "bug_summary": item["bug_summary"],
        "label_plan": item["planner_text"]
    })

# Save
json.dump(multi, open(OUT_PATH, "w"), indent=2, ensure_ascii=False)
print("Saved →", OUT_PATH)
