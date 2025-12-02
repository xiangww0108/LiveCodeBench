import json
import subprocess
import tempfile
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


# ===== CONFIG =====
MODEL_PATH = "data/output"  # your multitask-debugger model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BENCHMARK_FILE = "benchmark.json"   # you put benchmark problems here


# ===== UTILITIES =====

def extract_python_code(text):
    """
    Extract python code from ```python ... ```
    If none found, return whole text.
    """
    if "```" in text:
        m = re.search(r"```python(.*?)```", text, re.S)
        if m:
            return m.group(1).strip()
    return text.strip()


def run_code(code, input_str):
    """
    Run code in isolated subprocess, capture stdout.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name

    try:
        proc = subprocess.run(
            ["python3", fname],
            input=input_str.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3
        )
        stdout = proc.stdout.decode().strip()
        stderr = proc.stderr.decode().strip()
        return stdout, stderr
    except Exception as e:
        return "", str(e)


# ===== MAIN EVALUATION =====

def evaluate():
    print(f"Loading debugger model: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE
    )

    benchmark = json.load(open(BENCHMARK_FILE))
    total = len(benchmark)
    passed = 0

    for i, prob in enumerate(benchmark):
        print(f"\n===== Problem {i+1}/{total}: {prob['question_title']} =====")

        question = prob["question_content"]
        buggy_code = prob["buggy_code"]
        hidden_tests = prob["hidden_tests"]   # list of (input, expected_output)

        # ===== Construct debugging prompt =====
        messages = [
            {"role": "system", "content": "You are an expert code debugger."},
            {"role": "user", "content":
                f"PROBLEM:\n{question}\n\n"
                f"BUGGY CODE:\n{buggy_code}\n\n"
                "Fix the bug. Output only Python code inside a ```python``` block."
            }
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

        # ===== MODEL GENERATE =====
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        fixed_code = extract_python_code(response)

        print("---- Fixed Code Preview ----")
        print(fixed_code[:200])

        # ===== RUN HIDDEN TESTS =====
        all_pass = True
        for inp, expected in hidden_tests:
            out, err = run_code(fixed_code, inp)
            if err or out.strip() != expected.strip():
                all_pass = False
                print(f"❌ Test Failed. Input={inp!r} | Output={out!r} | Expected={expected!r}")
                break

        if all_pass:
            passed += 1
            print("✅ PASS")
        else:
            print("❌ FAIL")

    # ===== SUMMARY =====
    print("\n======================")
    print(" Debug Model Evaluation Complete")
    print("======================")
    print(f"PASS@1 = {passed}/{total} = {passed/total:.4f}")


if __name__ == "__main__":
    evaluate()
