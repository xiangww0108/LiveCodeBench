import json


def build_prompt(sample):
    task = sample["task"]

    # -------------- Common fields --------------
    problem = sample.get("question_content", "")
    metadata = sample.get("metadata", [])
    metadata_str = metadata[0] if len(metadata) > 0 else ""
    code = sample["code_list"][0] if isinstance(sample.get("code_list"), list) else sample.get("code_list", "")
    numbered_code = "\n".join(f"{i+1}: {line}" for i, line in enumerate(code.splitlines()))

    # -------------- LOCALIZER --------------
    if task == "localizer":
        error = metadata_str  # failing test
        bug_span = sample["bug_span"]

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code debugger. Locate the bug. Output ONLY a JSON list of line ranges like [[start, end]]."
                },
                {
                    "role": "user",
                    "content": (
                        f"PROBLEM:\n{problem}\n\n"
                        f"ERROR:\n{error}\n\n"
                        f"CODE:\n{numbered_code}"
                    )
                },
                {
                    "role": "assistant",
                    "content": json.dumps(bug_span)
                }
            ]
        }

    # -------------- PLANNER --------------
    elif task == "planner":
        prompt = f"""You are a precise and concise bug fixer (planner only).

Given:
- Problem: {problem[:500]}...
- Buggy code:
{code}

- Metadata: {metadata_str}
- Bug spans: {sample['bug_span']}
- Bug summary: {sample['bug_summary']}

Your task: Produce a short sequence of steps (2-6) describing how to fix the bug. Include a suggested corrected code snippet.

Plan:"""

        return {
            "prompt": prompt,
            "target": sample["planner_text"]
        }

    else:
        raise ValueError(f"Unknown task: {task}")
