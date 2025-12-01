import json


def build_prompt(sample):
    task = sample["task"]
    problem = sample.get("question_content", "")
    code = sample["code_list"][0]
    metadata = sample.get("metadata", [])
    metadata_str = metadata[0] if metadata else ""
    numbered_code = "\n".join(f"{i+1}: {line}" for i, line in enumerate(code.splitlines()))

    # ---------- LOCALIZER ----------
    if task == "localizer":

        # For manually curated dataset
        bug_span = sample.get("label_bug_span")
        bug_summary = sample.get("label_bug_summary")

        prompt = f"""### Task: localizer
    ### Problem:
    {problem}

    ### Code:
    {numbered_code}

    ### Error Trace / Test Failure:
    {metadata_str}

    ### What to do:
    Identify the buggy line range(s) AND write a brief summary of the bug.
    Output a JSON dict with fields "bug_span" and "bug_summary".

    ### Answer:"""

        target = json.dumps({
            "bug_span": bug_span,
            "bug_summary": bug_summary
        })

        return prompt, target

    # ---------- PLANNER ----------
    elif task == "planner":
        bug_span = sample.get("bug_span", [])
        bug_summary = sample.get("bug_summary", "")
        planner_text = sample.get("label_plan", "")

        prompt = f"""### Task: planner
    ### Problem:
    {problem[:600]}

    ### Code:
    {code}

    ### Bug Span:
    {bug_span}

    ### Bug Summary:
    {bug_summary}

    ### What to do:
    Produce 2-6 steps explaining how to fix the bug and provide a corrected snippet.

    ### Answer:"""

        target = planner_text
        return prompt, target

    else:
        raise ValueError(f"Unknown task: {task}")
