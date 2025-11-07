import re, os, json, pandas as pd, numpy as np
from typing import Union, List, Dict, Any


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load records from a .json file"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path not found: {path}")
    if not path.endswith(".json"):
        raise ValueError("Expected .json file")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------- ProblemType 主分类 ----------
def assign_problem_type(row):
    text = (str(row.get("question_title", "")) + " " +
            str(row.get("question_content", ""))).lower()

    # ---- String: Pattern / Transform / Property ----
    if re.search(r'\b(string|substring|palindrome|anagram|reverse|concatenate|replace|remove|insert|split|rearrange|vowel|character|letters|words|text|mirror|capslock|prefix|suffix|pattern|subsequence)\b', text):
        # 排除纯输入类 string（如 “each string represents a record”）
        if not re.search(r'\b(number|array|matrix|count|sum|digit|game|simulation|process|age|citizen|value)\b', text):
            if re.search(r'\b(prefix|suffix|substring|pattern|find|match|occurrence)\b', text):
                return "String-Pattern"
            elif re.search(r'\b(replace|remove|insert|concatenate|rearrange|sort|transform|modify)\b', text):
                return "String-Transform"
            elif re.search(r'\b(palindrome|equal|lexicographically|upper|lower|case|mirror|vowel)\b', text):
                return "String-Property"
            else:
                return "String-Other"

    # ---- Data Structure ----
    if re.search(r'\b(tree|dfs|bfs|graph|node|edge|parent|child|root)\b', text):
        return "DataStructure-GraphTree"
    if re.search(r'\b(stack|push|pop|bracket|parentheses)\b', text):
        return "DataStructure-Stack"
    if re.search(r'\b(queue|enqueue|dequeue|line|waiting)\b', text):
        return "DataStructure-Queue"

    # ---- Arithmetic ----
    if re.search(r'\b(array|matrix|list|sequence|subarray|subsequence|prefix|suffix)\b', text):
        return "Arithmetic-ArrayMatrix"
    if re.search(r'\b(sum|count|max|min|product|average|median|frequency|pairs|triplets|combination|permutation|number|digit|score|value|difference|range|interval)\b', text):
        return "Arithmetic-Counting"
    if re.search(r'\b(prime|gcd|lcm|divisible|mod|factor|multiple|remainder|bit|xor|mask|bitwise|shift|binary|boolean|logic)\b', text):
        return "Arithmetic-NumberTheory"

    # ---- Simulation ----
    if re.search(r'\b(simulate|simulation|process|step|turn|move|round|game|player|robot|machine|battle|fight|board|grid)\b', text):
        return "Simulation"

    # ---- Other ----
    return "Other"

import re

def classify_wrong_answer(msg: str) -> str:
    if not isinstance(msg, str) or not msg.strip():
        return "Pass"
    s = msg.lower().strip()
    if not s.startswith("wrong answer"):
        if "runtime" in s: return "Runtime Error"
        if "time limit" in s: return "Timeout"
        if "eof" in s or "reading a line" in s: return "Input Error"
        if "mismatched output length" in s: return "Length Mismatch"
        return "Other"

    if "!=" not in s:
        return "WrongAnswer (no diff shown)"
    left_right = s.split("!=")
    left, right = left_right[0], left_right[1]
    if re.search(r"\b\d+\b", left) and re.search(r"\b\d+\b", right):
        return "WrongAnswer-NumericMismatch"
    if re.search(r"\b(yes|no|true|false)\b", s):
        return "WrongAnswer-BooleanMismatch"
    if re.search(r"[-=+#<>]+", s) and len(re.findall(r"[-=+#<>]+", s)) > 1:
        return "WrongAnswer-SymbolMismatch"
    return "WrongAnswer-TextMismatch"

def classify_error_better(msg: str) -> str:
    if not isinstance(msg, str) or not msg.strip():
        return "Pass"
    s = msg.lower().strip()

    # wrong answer cases
    if s.startswith("wrong answer"):
        return classify_wrong_answer(msg)

    if "error during testing" in s:
        if "unpack" in s: return "TestingError-Unpacking"
        if "index out of range" in s: return "TestingError-Index"
        if "invalid syntax" in s: return "TestingError-Syntax"
        if "indented block" in s: return "TestingError-Indention"
        if "generator expression" in s: return "TestingError-Generator"
        return "TestingError-Other"

    if "time limit" in s: return "Timeout"
    if "memory limit" in s: return "MemoryError"
    if "runtime error" in s: return "RuntimeError"
    if "eof" in s or "reading a line" in s: return "Input Error"
    if "mismatched output length" in s: return "Length Mismatch"
    return "Other"

# ----------  Analysis ----------
import json

def extract_error_message(meta_list):
    if not isinstance(meta_list, list) or len(meta_list) == 0:
        return ""
    try:
        # 有些是字符串 JSON
        m = meta_list[0]
        if isinstance(m, str):
            obj = json.loads(m)
        elif isinstance(m, dict):
            obj = m
        else:
            return ""
        return obj.get("error_message", "")
    except Exception:
        return ""
    
def analyze(records: List[Dict[str, Any]]):
    
    df = pd.DataFrame(records)

    df["error_message_norm"] = df["metadata"].apply(extract_error_message)

    # ---- ProblemType ----
    df["ProblemType"] = df.apply(assign_problem_type, axis=1)

    # ---- ErrorType ----
    possible_err_keys = [c for c in df.columns if "error" in c.lower() or "message" in c.lower()]
    if possible_err_keys:
        key = possible_err_keys[0]
        df["ErrorType"] = df[key].apply(classify_error_better)
    else:
        print("⚠️ Warning: No error message column found. Filling ErrorType='Unknown'.")
        df["ErrorType"] = "Unknown"


    # ---- Aggregations ----
    problem_counts = df["ProblemType"].value_counts().reset_index()
    problem_counts.columns = ["ProblemType","count"]

    error_counts = df["ErrorType"].value_counts().reset_index()
    error_counts.columns = ["ErrorType","count"]

    diff_counts = (
        df.groupby("difficulty")["ProblemType"].count().reset_index()
        if "difficulty" in df.columns else pd.DataFrame()
    )
    if not diff_counts.empty:
        diff_counts.columns = ["difficulty","count"]

    plat_counts = (
        df.groupby("platform")["ProblemType"].count().reset_index()
        if "platform" in df.columns else pd.DataFrame()
    )
    if not plat_counts.empty:
        plat_counts.columns = ["platform","count"]

    return df, problem_counts, error_counts, diff_counts, plat_counts

