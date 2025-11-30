#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
localize_bug_spans.py — universal compatibility for OpenAI Responses API.
Tries: (1) new style response_format, (2) old style text.format with name, (3) older style text.format.schema.
"""
import argparse, json, time
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

def _load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list")
    return data

def _extract_primary_code(obj: Dict[str, Any]) -> Optional[str]:
    cl = obj.get("code_list") or []
    if cl:
        c0 = cl[0]
        if isinstance(c0, str) and c0.strip().startswith("```"):
            lines = c0.strip().splitlines()
            if len(lines)>=2 and lines[0].startswith("```") and lines[-1].startswith("```"):
                return "\n".join(lines[1:-1])
        return c0 if isinstance(c0, str) else None
    ol = obj.get("output_list") or []
    if ol:
        o0 = ol[0]
        if isinstance(o0, str) and o0.strip().startswith("```"):
            lines = o0.strip().splitlines()
            if len(lines)>=2 and lines[0].startswith("```") and lines[-1].startswith("```"):
                return "\n".join(lines[1:-1])
        return o0 if isinstance(o0, str) else None
    return None

def _extract_io_from_metadata(meta: List[Any]) -> Tuple[Optional[str],Optional[str],Optional[str]]:
    if not meta:
        return (None,None,None)
    b = meta[0]
    if isinstance(b,str):
        try: b = json.loads(b)
        except: b = {}
    if not isinstance(b, dict):
        return (None,None,None)
    return (b.get("inputs"), b.get("expected"), b.get("error_message") or b.get("error"))

def _mk_prompt(item, code: str, inputs, expected, error_message) -> str:
    numbered = _number_code(code)
    title = (item.get('question_title') or '').strip()

    return f"""
You are an overconfident yet precise bug localizer.
Your mission: locate the exact line(s) that caused the failure — no excuses, no hesitation.

Rules:
- The candidate code is shown with explicit 1-based line numbers in the left margin: "<LN> | <code>".
- Identify the SMALLEST contiguous line span (1-based inclusive) that most likely causes the failure.
- If multiple micro-spans are clearly required, return up to 2 spans (two [start,end] pairs).
- If you truly believe there is NO bug, return an empty list — otherwise NEVER leave it empty.
- Never explain your reasoning or show chain-of-thought.
- Respond as a strict JSON object with fields: "spans" and "summary". No surrounding text.

Output JSON schema:
{{
  "spans": [{{"start": <int>, "end": <int>}}, ...],   // 1-based inclusive; at most 2 items
  "summary": "<one-sentence concise description>"
}}

Context (brief problem title):
{title}

Candidate code (WITH line numbers):
<<<PY
{numbered}
PY

Failing context (one test):
inputs:
{inputs or ''}

expected_output:
{expected or ''}

error_message:
{error_message or ''}
"""

def _json_schema():
    return {
        "name": "BugSpans",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "integer","minimum":1},
                            "end":   {"type": "integer","minimum":1}
                        },
                        "required": ["start","end"],
                        "additionalProperties": False
                    },
                    "maxItems":2
                },
                "summary": {"type":"string","maxLength":160}
            },
            "required":["spans"],
            "additionalProperties":False
        }
    }

def _extract_text(resp) -> str:
    try:
        return resp.output_text
    except Exception:
        pass
    try:
        # 某些版本是 list 结构
        return resp.output[0].content[0].text
    except Exception:
        return ""

def _number_code(src: str) -> str:
    lines = src.splitlines()
    # keep width aligned (e.g., 1..999 -> width 3)
    w = max(2, len(str(len(lines))))
    return "\n".join(f"{str(i+1).rjust(w)} | {ln}" for i, ln in enumerate(lines))

def _call_openai(client, model: str, prompt: str) -> Dict[str, Any]:
    """
    Chat Completions + function calling (tools) for structured spans.
    Two-pass strategy:
      - pass#1: normal instruction
      - if spans == [] but summary suggests a bug, pass#2: force at least one span
    """
    import json

    parameters = {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "minimum": 1},
                        "end": {"type": "integer", "minimum": 1}
                    },
                    "required": ["start", "end"],
                    "additionalProperties": False
                },
                "maxItems": 2
            },
            "summary": {"type": "string", "maxLength": 160}
        },
        "required": ["spans", "summary"],
        "additionalProperties": False
    }

    def _once(ptext: str) -> Dict[str, Any]:
        # 1) 调用 API
        try:
            comp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise bug localizer. "
                            "You must call the function `return_spans` to return the result. "
                            "Do NOT print JSON directly; instead, call the function. "
                            "If you cannot find a bug, call the function with an empty list for spans."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            ptext
                            + "\n\nWhen you find the buggy lines, call:\n"
                              "return_spans({\n"
                              "  \"spans\": [{\"start\": X, \"end\": Y}],\n"
                              "  \"summary\": \"brief bug reason\"\n"
                              "})"
                        )
                    },
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "return_spans",
                            "description": "Return buggy spans (1-based inclusive line numbers) and a short summary.",
                            "parameters": parameters,
                        },
                    }
                ],
                tool_choice={"type": "function", "function": {"name": "return_spans"}},
                temperature=0.0,
            )
        except Exception as e:
            return {"spans": [], "summary": f"Request failed: {type(e).__name__}: {e}"}

        # 2) 解析 tool call
        try:
            tc = comp.choices[0].message.tool_calls
        except Exception:
            tc = None

        # 3) 如果没有触发工具，尝试从纯文本 JSON 兜底解析（有些模型/SDK会直接吐文本）
        if not tc:
            try:
                content = comp.choices[0].message.content or ""
            except Exception:
                content = ""
            try:
                data = json.loads(content)
            except Exception:
                # 尝试最小兜底
                return {"spans": [], "summary": "No function call triggered and no valid JSON content."}
        else:
            try:
                args_json = tc[0].function.arguments
                data = json.loads(args_json)
            except Exception as e:
                return {"spans": [], "summary": f"Parse error: {e}"}

        # 4) 统一归一化为 list-of-lists（[[a,b], ...]）
        spans = data.get("spans") or []
        norm = []
        for s in spans:
            if isinstance(s, dict) and "start" in s and "end" in s:
                a, b = int(s["start"]), int(s["end"])
            elif isinstance(s, (list, tuple)) and len(s) == 2:
                a, b = int(s[0]), int(s[1])
            else:
                continue
            if a > b:
                a, b = b, a
            norm.append([a, b])

        return {"spans": norm, "summary": (data.get("summary") or "")[:160]}

    # pass #1
    r1 = _once(prompt)

    # summary implies a bug?
    summary_lower = (r1.get("summary") or "").lower()
    looks_buggy = any(k in summary_lower for k in [
        "bug", "wrong", "incorrect", "fix", "error", "exception", "runtime", "wa", "mismatch", "zero division"
    ])

    if r1["spans"] or not looks_buggy:
        return r1

    # pass #2: force at least one span
    forced_prompt = (
        prompt
        + "\n\nIMPORTANT: Do NOT return an empty spans list. "
          "If uncertain, choose the single most suspicious line as [start==end]."
    )
    r2 = _once(forced_prompt)
    if not r2["spans"]:
        r1["summary"] = (r1.get("summary") or "") + " [fallback: model refused to return spans]"
        return r1
    return r2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True, help="Input JSON path")
    parser.add_argument("--out", dest="outp", required=True, help="Output JSON path")
    parser.add_argument("--model", default="gpt-5", help="Model name")
    parser.add_argument("--rate_limit_per_min", type=int, default=60, help="Requests per minute")
    args = parser.parse_args()

    if OpenAI is None:
        raise SystemExit("Please install openai: pip install openai")

    client = OpenAI()
    items = _load_items(args.inp)
    out_list = []
    wait = 60.0 / max(1, args.rate_limit_per_min)

    for obj in items:
        code = _extract_primary_code(obj)
        inputs, expected, err = _extract_io_from_metadata(obj.get("metadata", []))
        if not isinstance(code, str):
            obj["bug_span"] = []
            obj["bug_summary"] = "No candidate code found"
            out_list.append(obj)
        else:
            prompt = _mk_prompt(obj, code, inputs, expected, err)
            result = _call_openai(client, args.model, prompt)
            spans = result.get("spans", [])
            norm = []
            for s in spans:
                if isinstance(s, dict) and "start" in s and "end" in s:
                    a, b = int(s["start"]), int(s["end"])
                elif isinstance(s, (list, tuple)) and len(s) == 2:
                    a, b = int(s[0]), int(s[1])
                else:
                    continue
                if a > b:
                    a, b = b, a
                norm.append([a, b])
            obj["bug_span"] = norm
            obj["bug_summary"] = result.get("summary")
            out_list.append(obj)
        time.sleep(wait)

    with open(args.outp, "w", encoding="utf-8") as f2:
        json.dump(out_list, f2, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
