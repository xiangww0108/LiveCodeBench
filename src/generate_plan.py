#!/usr/bin/env python3
"""
Generate planner labels using an API (e.g., OpenAI),
based on training data that already has bug_span and bug_summary.

- Input : train-small.json
- Output: train-post.json with planner_text
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI   # pip install openai

TRAIN_PATH = Path("output/Qwen3-TrainTest-data/train-pre.json")
OUT_PATH   = Path("output/Qwen3-TrainTest-data/train-post.json")

MODEL_NAME = "gpt-4.1-mini"


# -------------------------------------------------------
# Build test snippet (optional, included in metadata)
# -------------------------------------------------------
def build_test_snippet(metadata_list: List[str]) -> str:
    if not metadata_list:
        return "No tests available."

    try:
        m = json.loads(metadata_list[0])
    except Exception:
        return "\n".join(metadata_list)

    parts = []
    if "inputs" in m:
        parts.append("Inputs:\n" + str(m["inputs"]).strip())
    if "expected" in m:
        parts.append("Expected:\n" + str(m["expected"]).strip())
    if "output" in m:
        parts.append("Model Output:\n" + str(m["output"]).strip())
    if "error_message" in m:
        parts.append("Error:\n" + str(m["error_message"]).strip())

    return "\n\n".join(parts) if parts else json.dumps(m, ensure_ascii=False)


# -------------------------------------------------------
# FINAL Planner Prompt (ONE-SHOT + real task)
# -------------------------------------------------------
def build_planner_prompt(
    question_content: str,
    code_list: str,
    metadata: str,
    bug_span: Any,
    bug_summary: str
) -> str:

    bug_span_json = json.dumps(bug_span, ensure_ascii=False)

    # Use .format() → SAFE for {}, JSON, code blocks
    return """
You are a precise and concise bug fixer (planner only).

Given:
- problem description
- buggy candidate code
- metadata (inputs, expected output, error)
- bug spans
- bug summary

Your task:
Produce ONE output block:
A short sequence of steps (2-6) describing how to fix the bug.
Inside the plan, include a **suggested corrected code snippet**.
DO NOT output a unified diff.
DO NOT output anything outside.

Rules:
- Be minimal and target only the bug_span lines unless necessary.
- Provide a tiny patch snippet in plain Python.
- No chain of thought.
- No extra commentary.

====================================================
ONE-SHOT EXAMPLE
====================================================
INPUT:
<bug_summary>
The function returns the wrong comparison operator.
</bug_summary>

<code_list>
def f(x):
    return x > 10
</code_list>

<bug_span>
[[2,2]]
</bug_span>

OUTPUT:
1. Line 2 uses '>' but the correct comparison is '>='.
2. Replace the return condition with the corrected operator.

Suggested patch:
    return x >= 10

====================================================
NOW THE REAL TASK
====================================================
===== INPUT =====
<code_list>
{code_list}
</code_list>

<bug_span>
{bug_span_json}
</bug_span>

<bug_summary>
{bug_summary}
</bug_summary>

===== OUTPUT =====
...
""".format(
        question_content=question_content,
        code_list=code_list,
        metadata=metadata,
        bug_span_json=bug_span_json,
        bug_summary=bug_summary
    )


# -------------------------------------------------------
# OpenAI API call
# -------------------------------------------------------
def call_planner_api(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert debugging system."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# -------------------------------------------------------
# Main processing loop
# -------------------------------------------------------
def main():

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("Missing environment variable: OPENAI_API_KEY")

    client = OpenAI()

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Input not found: {TRAIN_PATH}")

    data = json.loads(TRAIN_PATH.read_text())
    print(f"[INFO] Loaded {len(data)} samples.")

    out = []

    for idx, item in enumerate(data):

        # already generated
        if item.get("planner_text"):
            out.append(item)
            continue

        # Extract fields from JSON
        question_content = item.get("question_content", "")
        code_list_raw = item.get("code_list") or []
        code_list = code_list_raw[0] if code_list_raw else ""

        metadata_list = item.get("metadata") or []
        metadata = metadata_list[0] if metadata_list else "{}"

        bug_span = item.get("bug_span") or []
        bug_summary = item.get("bug_summary", "")

        # Build prompt
        prompt = build_planner_prompt(
            question_content=question_content,
            code_list=code_list,
            metadata=metadata,
            bug_span=bug_span,
            bug_summary=bug_summary
        )

        # API call
        try:
            planner_text = call_planner_api(client, prompt)
        except Exception as e:
            print(f"[ERROR] idx={idx}, qid={item.get('question_id')}: {e}")
            item["planner_text"] = ""
            out.append(item)
            continue

        item["planner_text"] = planner_text
        out.append(item)

        if (idx + 1) % 20 == 0:
            print(f"[INFO] processed {idx+1}/{len(data)}")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[DONE] saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
