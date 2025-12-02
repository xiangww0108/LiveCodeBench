import json
import os
import torch
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ================= CONFIGURATION =================
LOCALIZER_ID = "Intellegen4/Qwen2.5-Coder-1.5B-Localizer-FullSFT"
PLANNER_ID = "Intellegen4/modular-planner-qwen2.5-coder-1.5b"
FIXER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def add_line_numbers(code):
    """Adds line numbers to the code for the Localizer."""
    lines = code.split('\n')
    return "\n".join([f"{i+1} | {line}" for i, line in enumerate(lines)])

def parse_span(text):
    """
    Robustly extracts the FIRST JSON list of lists from text (e.g., [[1, 3]]).
    Adapted from structured_pipeline/localizer/evaluate.py
    """
    try:
        match = re.search(r"\[\[.*?\]\]", text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return [[1, 1]] # Default fallback

def safe_load_problems():
    print("Loading problems from Hugging Face (Parquet mode)...")
    try:
        # Try loading the main dataset which is usually parquet-based now
        dataset = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Standard load failed ({e}). Attempting direct Parquet load...")
        # Fallback: Load the full version which might be updated
        dataset = load_dataset("livecodebench/code_generation", split="test")
        
    # Convert to a dictionary map directly
    problem_map = {}
    for item in dataset:
        # Adjust keys based on actual dataset structure
        q_id = item.get("question_id")
        if q_id:
            problem_map[q_id] = item
    return problem_map

class ModularPipeline:
    def __init__(self):
        print(f"[Init] Loading Localizer: {LOCALIZER_ID}")
        self.loc_tokenizer = AutoTokenizer.from_pretrained(LOCALIZER_ID)
        self.loc_model = AutoModelForCausalLM.from_pretrained(
            LOCALIZER_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.loc_model.eval()

        print(f"[Init] Loading Planner: {PLANNER_ID}")
        self.plan_tokenizer = AutoTokenizer.from_pretrained(PLANNER_ID, trust_remote_code=True)
        if self.plan_tokenizer.pad_token is None:
            self.plan_tokenizer.pad_token = self.plan_tokenizer.eos_token
        self.plan_model = AutoModelForCausalLM.from_pretrained(
            PLANNER_ID, 
            torch_dtype=torch.float16, 
            device_map="auto", 
            trust_remote_code=True
        )
        self.plan_model.eval()

        print(f"[Init] Loading Fixer: {FIXER_ID}")
        self.fixer_tokenizer = AutoTokenizer.from_pretrained(FIXER_ID)
        self.fixer_model = AutoModelForCausalLM.from_pretrained(
            FIXER_ID, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        self.fixer_model.eval()

    def run_localizer(self, problem_text, code, error):
        """
        Locates the bug in the code.
        Prompt format adapted from structured_pipeline/localizer/evaluate.py
        """
        numbered_code = add_line_numbers(code)
        messages = [
            {"role": "system", "content": "You are a code debugger. Locate the bug. Output ONLY a JSON list of line ranges like [[start, end]]."},
            {"role": "user", "content": f"PROBLEM:\n{problem_text}\n\nERROR:\n{error}\n\nCODE:\n{numbered_code}"}
        ]
        
        text = self.loc_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.loc_tokenizer(text, return_tensors="pt").to(self.loc_model.device)
        
        with torch.no_grad():
            outputs = self.loc_model.generate(
                **inputs, 
                max_new_tokens=50, 
                temperature=0.1, 
                do_sample=False
            )
        
        response = self.loc_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return parse_span(response)

    def run_planner(self, problem_text, code, error, bug_span):
        """
        Generates a repair plan.
        Prompt format adapted from structured_pipeline/planner/inference_planner.py
        """
        # Truncate question if too long (as done in the source script)
        if len(problem_text) > 1000:
            problem_text = problem_text[:1000] + "..."

        # We treat the error message as metadata/bug summary for the planner
        prompt = f"""You are a precise and concise bug fixer (planner only).

Given:
- Problem: {problem_text}
- Buggy code:
{code}

- Metadata: {error}
- Bug spans: {bug_span}
- Bug summary: {error}

Your task: Produce a short sequence of steps (2-6) describing how to fix the bug. Include a suggested corrected code snippet.

Plan:"""
        
        inputs = self.plan_tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096
        ).to(self.plan_model.device)
        
        with torch.no_grad():
            outputs = self.plan_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.plan_tokenizer.pad_token_id,
                eos_token_id=self.plan_tokenizer.eos_token_id
            )
        
        plan = self.plan_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return plan.strip()

    def run_fixer(self, problem_text, code, error, plan):
        """
        Generates the fixed code based on the plan.
        """
        messages = [
            {"role": "system", "content": "You are a competitive programmer. Fix the buggy code based on the provided plan. Return ONLY the corrected code block."},
            {"role": "user", "content": f"""PROBLEM:
{problem_text}

BUGGY CODE:
{code}

ERROR:
{error}

REPAIR PLAN:
{plan}

Please output the fixed Python code:"""}
        ]
        
        text = self.fixer_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.fixer_tokenizer(text, return_tensors="pt").to(self.fixer_model.device)
        
        with torch.no_grad():
            outputs = self.fixer_model.generate(
                **inputs, 
                max_new_tokens=1024, 
                temperature=0.2, 
                do_sample=True
            )
            
        response = self.fixer_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up code block markdown
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)
            
        return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to baseline output json")
    parser.add_argument("--output_file", type=str, default="modular_repair_output.json")
    parser.add_argument("--eval_file", type=str, default=None)
    args = parser.parse_args()

    # 1. Load Problems using the SAFE loader
    problem_map = safe_load_problems()
    print(f"Loaded {len(problem_map)} problems.")

    # 2. Load Data
    print(f"Loading input: {args.input_file}")
    with open(args.input_file, "r") as f:
        baseline_results = json.load(f)

    # 3. Init Pipeline
    pipeline = ModularPipeline()
    repaired_results = []

    print(f"Processing {len(baseline_results)} samples...")
    
    for i, entry in tqdm(enumerate(baseline_results), total=len(baseline_results)):
        q_id = entry["question_id"]
        
        # Check if question exists in our map
        if q_id not in problem_map:
            continue
            
        # Get content from the loaded map
        # Note: Accessing fields might differ if not using the custom class wrapper
        prob_item = problem_map[q_id]
        question_text = prob_item.get("question_content") or prob_item.get("description")
        
        # Get the code (assuming pass@1, taking the first sample)
        if isinstance(entry["code_list"], list):
            code = entry["code_list"][0]
        else:
            code = entry["code_list"]

        # Retrieve Error Message
        # If metadata exists and has error_message, use it. Otherwise generic.
        error_msg = "Wrong Answer or Runtime Error on hidden test cases."
        if "metadata" in entry and entry["metadata"]:
             meta = entry["metadata"]
             if isinstance(meta, dict) and "error_message" in meta:
                 error_msg = meta["error_message"]
             elif isinstance(meta, list) and len(meta) > 0 and isinstance(meta[0], dict):
                 error_msg = meta[0].get("error_message", error_msg)

        # --- STEP 1: LOCALIZE ---
        try:
            span = pipeline.run_localizer(question_text, code, error_msg)
            # Convert list to string format for the planner if needed, or keep as list
            span_str = str(span)
        except Exception as e:
            print(f"[Error] Localizer failed for {q_id}: {e}")
            span_str = "[[1, 10]]"

        # --- STEP 2: PLAN ---
        try:
            plan = pipeline.run_planner(question_text, code, error_msg, span_str)
        except Exception as e:
            print(f"[Error] Planner failed for {q_id}: {e}")
            plan = "Fix the logic error based on the problem description."

        # --- STEP 3: FIX ---
        try:
            fixed_code = pipeline.run_fixer(question_text, code, error_msg, plan)
            
            # Update entry
            # We keep the original structure but update code_list with the fixed version
            new_entry = entry.copy()
            new_entry["code_list"] = [fixed_code]
            new_entry["repair_info"] = {
                "original_error": error_msg,
                "bug_span": span_str,
                "repair_plan": plan
            }
            repaired_results.append(new_entry)
            
        except Exception as e:
            print(f"[Error] Fixer failed for {q_id}: {e}")
            repaired_results.append(entry) # Keep original if fix fails

        # Periodic save
        if (i + 1) % 10 == 0:
            with open(args.output_file, "w") as f:
                json.dump(repaired_results, f, indent=2)

    # Final Save
    with open(args.output_file, "w") as f:
        json.dump(repaired_results, f, indent=2)
    print(f"Repair complete. Saved to {args.output_file}")

if __name__ == "__main__":
    main()
    
