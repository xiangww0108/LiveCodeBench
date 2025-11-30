#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, re

FENCE_RE = re.compile(r"^```[\w-]*\n?(.*?)```$", re.DOTALL)

def looks_true(x):
    if isinstance(x, bool):
        return x is True
    if isinstance(x, (int, float)):
        return x == 1
    if isinstance(x, str):
        return x.strip().lower() in {"true", "1", "yes", "y"}
    return False

def has_any_true(lst):
    try:
        return any(looks_true(v) for v in lst)
    except Exception:
        return False

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    m = FENCE_RE.match(s)
    if m:
        s = m.group(1).strip()
    return s

def has_meaningful_code(code_list) -> bool:
    """True iff code_list contains at least one non-empty code snippet."""
    if not isinstance(code_list, list):
        return False
    for item in code_list:
        if isinstance(item, str):
            inner = _strip_code_fence(item)
            if inner.strip():   # something besides whitespace/fences
                return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Remove entries whose graded_list contains true or code_list has no meaningful code.")
    ap.add_argument("--in", dest="inp", required=True, help="input JSON path (list of objects)")
    ap.add_argument("--out", dest="outp", required=True, help="output JSON path")
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise SystemExit("Input JSON must be a list of objects.")

    filtered = []
    for obj in data:
        # 1) Must have meaningful code_list
        if not has_meaningful_code(obj.get("code_list")):
            continue

        # 2) Drop if graded_list contains any truthy value
        gl = obj.get("graded_list", None)
        if gl is None:
            keep = True
        else:
            if not isinstance(gl, list):
                gl = [gl]
            keep = not has_any_true(gl)

        if keep:
            filtered.append(obj)

    with open(args.outp, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
