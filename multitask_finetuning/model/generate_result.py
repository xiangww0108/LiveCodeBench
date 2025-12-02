#!/usr/bin/env python3
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# ---------------------------
# Load JSON
# ---------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


# ----------- JSON extractor -------------

def extract_json(raw):
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        return json.loads(match.group(0))
    except:
        return None



# -----------------------------------------------------
#  LOCALIZER PROMPT (MESSAGES, CHAT FORMAT)
# -----------------------------------------------------
def make_localizer_messages(example):
    raw_code = example["code_list"][0]
    code_lines = raw_code.split("\n")
    numbered_code = "\n".join(
        [f"{i + 1} | {line}" for i, line in enumerate(code_lines)]
    )

    error = "Unknown Error"
    if "metadata" in example and example["metadata"]:
        try:
            meta_str = example["metadata"][0]
            meta_json = json.loads(meta_str)
            error = meta_json.get("error_message", meta_json.get("error", "Unknown Error"))
        except:
            pass

    messages = [
        {
            "role": "system",
            "content": (
                "You are a code bug localizer. "
                "Return ONLY a JSON dict with two keys:\n"
                "  - \"bug_span\": list of [start,end] line ranges\n"
                "  - \"bug_summary\": short English explanation\n\n"
                "Example:\n"
                "{\n"
                "  \"bug_span\": [[8,10]],\n"
                "  \"bug_summary\": \"Loop range too small\"\n"
                "}\n"
                "Do NOT return anything except JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"PROBLEM:\n{example['question_content']}\n\n"
                f"ERROR:\n{error}\n\n"
                f"CODE:\n{numbered_code}\n\n"
                "Return ONLY the JSON dict."
            ),
        },
    ]
    return messages

# -----------------------------------------------------
#  Chat-style generation (for localizer)
# -----------------------------------------------------
def generate_chat(model, tokenizer, messages):
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


# -----------------------------------------------------
#  PLANNER PROMPT (non-chat, plain text)
# -----------------------------------------------------
def make_planner_prompt(problem, code, bug_span, bug_summary):
    return f"""### Task: planner
### Problem:
{problem}

### Code:
{code}

### Bug Span:
{bug_span}

### Bug Summary:
{bug_summary}

### What to do:
Produce 2-6 clear steps explaining how to fix the bug and then provide a corrected code snippet inside a Markdown code block.

### Answer:
"""

def generate(model, tokenizer, prompt, max_new=800):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=max_new,
        temperature=0.0,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    gen_text = full_text[len(prompt):].strip()
    return gen_text


def extract_fixed_code(plan_output):
    # 1. Try code block first
    match = re.search(r"```(?:python)?\s*([\s\S]*?)```", plan_output)
    if match:
        return match.group(1).strip()

    # 2. Fallback: grab last indented code region
    lines = plan_output.split("\n")
    code_lines = []
    for line in lines:
        if line.startswith("    ") or line.startswith("\t") or "def " in line or line.strip().startswith("for ") or line.strip().startswith("while "):
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()

    return None

# -----------------------------------------------------
#  MAIN PIPELINE
# -----------------------------------------------------
def main():
    model_path = "data/output_new/checkpoint-200"
    test_file = "data/Qwen3-TrainTest-data/test-pre.json"
    output_file = (
        "data/Qwen3-TrainTest-data/multitask-step-by-step/finetuning_results.json"
    )

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    test_data = load_json(test_file)
    results = []

    print(f"Loaded {len(test_data)} test samples.")

    for idx, example in enumerate(test_data):
        print(f"\n=== Running sample {idx+1}/{len(test_data)} ===")

        problem = example["question_content"]
        code = example["code_list"][0]

        # -------- LOCALIZER ----------
        messages = make_localizer_messages(example)
        # messages.append({"role": "assistant", "content": ""})
        loc_output = generate_chat(model, tokenizer, messages)
        print("\n=== RAW LOCALIZER OUTPUT ===")
        print(loc_output)


        loc_json = extract_json(loc_output)
        if loc_json is not None:
            bug_span = loc_json.get("bug_span", [])
            bug_summary = loc_json.get("bug_summary", "")
        else:
            bug_span = []
            bug_summary = "parse_error"

        # -------- PLANNER -----------
        plan_prompt = make_planner_prompt(problem, code, bug_span, bug_summary)
        plan_output = generate(model, tokenizer, plan_prompt)
        print("\n=== RAW PLANNER OUTPUT ===")
        print(plan_output)
        fixed_code = extract_fixed_code(plan_output)
        print("\n=== EXTRACTED FIXED CODE ===")
        print(fixed_code)


        results.append(
            {
                "question_title": example["question_title"],
                "bug_span": bug_span,
                "bug_summary": bug_summary,
                "planner_output": plan_output,
                "fixed_code": fixed_code,
            }
        )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved generated results to {output_file}")


if __name__ == "__main__":
    main()
