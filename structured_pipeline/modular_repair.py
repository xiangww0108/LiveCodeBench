import json
import os
import torch
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= CONFIGURATION =================
# Models
LOCALIZER_ID = "Intellegen4/Qwen2.5-Coder-1.5B-Localizer-FullSFT"
PLANNER_ID = "Intellegen4/modular-planner-qwen2.5-coder-1.5b"
FIXER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" 

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def add_line_numbers(code):
    """Adds line numbers to the code for the Localizer."""
    if not code:
        return ""
    lines = code.split('\n')
    return "\n".join([f"{i+1} | {line}" for i, line in enumerate(lines)])

def parse_span(text):
    """
    Robustly extracts the FIRST JSON list of lists from text (e.g., [[1, 3]]).
    """
    try:
        match = re.search(r"\[\[.*?\]\]", text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return [[1, 1]] # Default fallback

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
        Stage 1: Localizer
        Identifies the lines of code responsible for the error.
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
        Stage 2: Planner
        Generates a step-by-step plan to fix the bug using the localized span and error info.
        """
        # Truncate question if too long to fit in context
        if len(problem_text) > 1500:
            problem_text = problem_text[:1500] + "..."

        prompt = f"""You are a precise and concise bug fixer (planner only).

Given:
- Problem: {problem_text}
- Buggy code:
{code}

- Metadata: {error}
- Bug spans: {bug_span}
- Bug summary: {error}

Your task: Produce a short sequence of steps describing how to fix the bug.

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
        Stage 3: Fixer
        Uses the Plan as 'Teacher Instructions' to rewrite the code.
        """
        messages = [
            {"role": "system", "content": "You are an expert programmer. Your task is to fix the buggy code provided below, following the specific instructions given by the teacher."},
            {"role": "user", "content": f"""PROBLEM DESCRIPTION:
{problem_text}

BUGGY CODE:
{code}

ERROR MESSAGE:
{error}

TEACHER INSTRUCTIONS (PLAN):
{plan}

Based on the instructions above, please output the complete, corrected Python code. Wrap the code in ```python ... ``` blocks."""}
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
        
        # Extract code block
        code_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)
        code_match = re.search(r"```\n(.*?)```", response, re.DOTALL)
        if code_match:
            return code_match.group(1)
            
        return response

def parse_metadata_error(metadata_field):
    """
    Parses the error message from the metadata list.
    Handles: ['{"error_message": "..."}']
    """
    default_error = "Wrong Answer on hidden test cases."
    
    if not metadata_field:
        return default_error
        
    try:
        # If it's a list, look at the first item
        if isinstance(metadata_field, list) and len(metadata_field) > 0:
            item = metadata_field[0]
            # If the item is a string (JSON dump), parse it
            if isinstance(item, str):
                try:
                    data = json.loads(item)
                    return data.get("error_message", default_error)
                except:
                    return item # Return raw string if not JSON
            # If the item is already a dict
            elif isinstance(item, dict):
                return item.get("error_message", default_error)
                
    except Exception as e:
        pass
        
    return default_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to evaluation results json")
    parser.add_argument("--output_file", type=str, default="repaired_results.json")
    args = parser.parse_args()

    # 1. Load Input Data
    print(f"Loading results from: {args.input_file}")
    with open(args.input_file, "r") as f:
        results = json.load(f)

    # 2. Init Pipeline
    pipeline = ModularPipeline()
    
    repaired_count = 0
    skipped_count = 0
    
    print(f"Processing {len(results)} samples...")
    
    for i, entry in tqdm(enumerate(results), total=len(results)):
        
        # --- Check if repair is needed ---
        # If pass@1 is 1.0, the code is already correct.
        if entry.get("pass@1") == 1.0:
            skipped_count += 1
            continue

        q_id = entry.get("question_id", f"unknown_{i}")
        
        # Extract Problem Content (Prioritize existing field)
        question_text = entry.get("question_content", "")
        if not question_text:
            # Fallback to question_title if content is missing (unlikely based on your sample)
            question_text = entry.get("question_title", "")

        # Extract Code
        code_list = entry.get("code_list", [])
        if isinstance(code_list, list) and len(code_list) > 0:
            code = code_list[0]
        elif isinstance(code_list, str):
            code = code_list
        else:
            code = ""

        # Skip empty code (cannot repair nothing)
        if not code or not code.strip():
            # print(f"Skipping {q_id}: Empty code.")
            continue

        # Extract Error
        error_msg = parse_metadata_error(entry.get("metadata"))

        # --- EXECUTE PIPELINE ---
        
        # 1. Localize
        try:
            span = pipeline.run_localizer(question_text, code, error_msg)
            span_str = str(span)
        except Exception as e:
            print(f"[{q_id}] Localizer error: {e}")
            span_str = "[[1, 100]]" # Fallback

        # 2. Plan (Teacher Instruction)
        try:
            plan = pipeline.run_planner(question_text, code, error_msg, span_str)
        except Exception as e:
            print(f"[{q_id}] Planner error: {e}")
            plan = "Debug the code based on the error message."

        # 3. Fix (Rewrite)
        try:
            fixed_code = pipeline.run_fixer(question_text, code, error_msg, plan)
            
            # Update the entry with the new code
            # Note: We replace code_list with the new fixed code for the next eval round
            entry["code_list"] = [fixed_code]
            
            # Store repair info for analysis
            entry["repair_info"] = {
                "original_error": error_msg,
                "bug_span": span_str,
                "teacher_plan": plan
            }
            
            # Reset metrics since we changed the code
            entry["pass@1"] = 0.0 # Will be re-evaluated later
            entry["graded_list"] = [] 
            entry["metadata"] = [] # Clear old error metadata
            
            repaired_count += 1
            
        except Exception as e:
            print(f"[{q_id}] Fixer error: {e}")

        # Periodic Save
        if (i + 1) % 10 == 0:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final Save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Repair process complete.")
    print(f"Repaired: {repaired_count}")
    print(f"Skipped (Already Correct): {skipped_count}")
    print(f"Saved to {args.output_file}")

if __name__ == "__main__":
    main()
    