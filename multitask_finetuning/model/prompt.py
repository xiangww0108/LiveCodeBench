import json


def build_prompt_localizer(sample):
    problem = sample["question_content"]
    raw_code = sample["code_list"][0]
    code_lines = raw_code.split("\n")
    numbered_code = "\n".join(
        f"{i+1} | {line}" for i, line in enumerate(code_lines)
    )

    error = "Unknown Error"
    if "metadata" in sample and sample["metadata"]:
        try:
            meta = json.loads(sample["metadata"][0])
            error = meta.get("error_message", meta.get("error", "Unknown Error"))
        except:
            pass

    # === ChatML messages ===
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
                f"PROBLEM:\n{problem}\n\n"
                f"ERROR:\n{error}\n\n"
                f"CODE:\n{numbered_code}\n\n"
                "Return ONLY the JSON dict."
            ),
        },
    ]

    # === target ===
    target = json.dumps({
        "bug_span": sample["label_bug_span"],
        "bug_summary": sample["label_bug_summary"]
    }, ensure_ascii=False)

    return messages, target

def build_prompt_planner(sample):
    problem = sample["question_content"]
    code = sample["code_list"][0]
    bug_span = sample.get("bug_span", [])
    bug_summary = sample.get("bug_summary", "")
    planner_text = sample.get("label_plan", "")

    prompt = f"""### Task: planner
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
Return plain text.

### Answer:
"""

    return prompt, planner_text

def build_prompt(sample):
    task = sample["task"]

    if task == "localizer":
        return build_prompt_localizer(sample)

    elif task == "planner":
        return build_prompt_planner(sample)

    else:
        raise ValueError("Unknown task")
