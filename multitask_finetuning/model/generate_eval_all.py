import json
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


# ======================================================
# Load multitask finetuned model
# ======================================================
MODEL_PATH = "data/output"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH, trust_remote_code=True, use_fast=False
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
).eval()

print("[INFO] Loaded multitask finetuned model.")


# ======================================================
# Chat generator (for localizer)
# ======================================================
def chat_generate(messages, max_new=200):
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# ======================================================
# Standard generator (for planner)
# ======================================================
def generate(prompt, max_new=400):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    return full[len(prompt):].strip()


# ======================================================
# Extract code from ``` blocks
# ======================================================
def extract_code(text):
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1]
    return None


# ======================================================
# Build multitask prompts
# ======================================================
def build_localizer_messages(item):
    raw_code = item["code_list"][0]
    code_lines = raw_code.split("\n")
    numbered = "\n".join([f"{i+1} | {line}" for i, line in enumerate(code_lines)])

    error = "Unknown Error"
    if item.get("metadata"):
        try:
            meta = json.loads(item["metadata"][0])
            error = meta.get("error_message", error)
        except:
            pass

    return [
        {
            "role": "system",
            "content": (
                "You are a code bug localizer.\n"
                "Return ONLY JSON with keys:\n"
                "- bug_span\n"
                "- bug_summary\n"
            )
        },
        {
            "role": "user",
            "content": (
                f"PROBLEM:\n{item['question_content']}\n\n"
                f"ERROR:\n{error}\n\n"
                f"CODE:\n{numbered}\n\n"
                "Return ONLY the JSON dict."
            )
        },
    ]


def build_planner_prompt(problem, code, bug_span, bug_summary):
    return f"""
### Task: planner
### Problem:
{problem}

### Code:
{code}

### Bug Span:
{bug_span}

### Bug Summary:
{bug_summary}

### What to do:
Provide 2–6 steps explaining the fix and then output corrected code inside ```python```.

### Answer:
"""


# ======================================================
# Extract JSON from localizer
# ======================================================
def extract_json(raw):
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        return json.loads(match.group(0))
    except:
        return None


# ======================================================
# Main inference: follow *YOUR TRAINED MULTITASK FORMAT*
# ======================================================
def run(input_path, save_path):
    data = json.load(open(input_path))
    results = []

    for item in tqdm(data):
        problem = item["question_content"]
        code = item["code_list"][0]

        # -------------------------
        # 1) LOCALIZER
        # -------------------------
        messages = build_localizer_messages(item)
        loc_raw = chat_generate(messages)
        loc_json = extract_json(loc_raw)

        if loc_json:
            bug_span = loc_json.get("bug_span", [])
            bug_summary = loc_json.get("bug_summary", "")
        else:
            bug_span = []
            bug_summary = "parse_error"

        # -------------------------
        # 2) PLANNER
        # -------------------------
        planner_prompt = build_planner_prompt(problem, code, bug_span, bug_summary)
        planner_output = generate(planner_prompt)

        fixed_code = extract_code(planner_output)

        results.append({
            "question_title": item["question_title"],
            "bug_span_pred": bug_span,
            "bug_summary_pred": bug_summary,
            "planner_output_pred": planner_output,
            "fixed_code_pred": fixed_code
        })

    with open(save_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] Saved inference to {save_path}")


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--save", required=True)
    args = parser.parse_args()
    run(args.input, args.save)
