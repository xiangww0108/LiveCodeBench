import json
import torch
import re
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= ⚙️ CONFIGURATION (CONSTANTS) ⚙️ =================
# 1. E2E L/P 模型权重所在的本地路径 
LOCALIZER_PLANNER_MODEL_PATH = "/home/ubuntu/finetune_e2e_new/model" 
# 2. 用于加载 Tokenizer 的基座模型 ID
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# 3. Fixer 模型 ID 
FIXER_ID = "Qwen/Qwen2.5-Coder-7B-Instruct" 
# 5. 默认输入文件路径 (如果不通过命令行 --input_file 指定)
DEFAULT_INPUT_FILE = "/home/ubuntu/finetune_e2e_new/data/Scenario.selfrepair_1_0.2_eval_all.json"
# 6. 默认输出文件路径 (如果不通过命令行 --output_file 指定)
DEFAULT_OUTPUT_FILE = "/home/ubuntu/finetune_e2e_new/data/Qwen2-1_5B_repaired_extrinsic.json_repaired_extrinsic.json"
# =================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ========================================
# 🔧 Helpers 
# ========================================

def add_line_numbers(code):
    """Adds line numbers to the code for the Localizer's input prompt."""
    if not code: return ""
    lines = code.split('\n')
    return "\n".join([f"{i+1} | {line}" for i, line in enumerate(lines)])

def extract_code_from_fixer(output):
    """Extracts code block from Fixer's output."""
    m = re.search(r"```python\n(.*?)```", output, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"```\n(.*?)```", output, re.DOTALL)
    if m: return m.group(1).strip()
    return output.strip()

def parse_metadata_error(metadata_field):
    """Parses the error message from the metadata list."""
    default_error = "Wrong Answer on hidden test cases."
    if not metadata_field: return default_error
    try:
        item = metadata_field[0] if isinstance(metadata_field, list) and len(metadata_field) > 0 else metadata_field
        if isinstance(item, str): 
            try: return json.loads(item).get("error_message", default_error)
            except: return item
        elif isinstance(item, dict): return item.get("error_message", default_error)
    except: pass
    return default_error

# -----------------------------------------------------------------
# 1. JSON 提取器
# -----------------------------------------------------------------
def extract_json(raw: str):
    """Uses regex to find and parse a JSON object in the raw output."""
    try:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match: return None
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    except Exception:
        return None

# ========================================
# ⚙️ Repair Pipeline (E2E Version)
# ========================================
class RepairPipeline:
    def __init__(self):
        # --- Localizer/Planner Model (E2E) ---
        print(f"[Init] Loading E2E L/P model from: {LOCALIZER_PLANNER_MODEL_PATH}")
        self.lp_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.lp_tokenizer.pad_token is None: self.lp_tokenizer.pad_token = self.lp_tokenizer.eos_token
        self.lp_model = AutoModelForCausalLM.from_pretrained(
            LOCALIZER_PLANNER_MODEL_PATH, 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            trust_remote_code=True
        ).eval()

        # --- Fixer Model ---
        print(f"[Init] Loading Fixer model: {FIXER_ID}")
        self.fixer_tokenizer = AutoTokenizer.from_pretrained(FIXER_ID, trust_remote_code=True)
        self.fixer_model = AutoModelForCausalLM.from_pretrained(
            FIXER_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()

    # ---------------- L/P Generator (E2E 合并 - 采用 SFT Prompt) ----------------
    def run_e2e_analysis(self, problem: str, code: str, error: str):
        """ Runs E2E model to generate bug_span, bug_summary, and planner_text in one go."""
        
        # SFT 训练配置和生成参数
        MAX_SEQ_LENGTH = 4096 
        
        generation_config = self.lp_model.generation_config
        generation_config.max_new_tokens = 1024 # 优化到 1024
        generation_config.do_sample = True 
        generation_config.temperature = 0.8 
        generation_config.top_p = 0.9 
        generation_config.top_k = 50  
        generation_config.eos_token_id = [
            self.lp_tokenizer.eos_token_id,              
            self.lp_tokenizer.convert_tokens_to_ids('}') 
        ] 
        
        # --- SYSTEM PROMPT ---
        system_msg = (
            """You are a strict 3-stage code analysis model.
        You must follow the stages in order and output ONLY the final JSON object.

        STAGE 1: Based on the buggy code -> Generate bug_span (A list of line ranges for the error, format: [start, end] lines).
        STAGE 2: Using the buggy code and bug_span -> Generate bug_summary (A concise summary of the bug in English).
        STAGE 3: Using the buggy code, bug_span, and bug_summary -> Generate planner_text (The step-by-step repair plan in English).

        RULES:
        - You MUST generate all 3 fields: bug_span, bug_summary, and planner_text.
        - The output MUST be, and ONLY be, a single JSON object.
        - All text output (summary and planner_text) MUST be in English.
        """
        )

        # --- USER PROMPT ---
        user_msg = (
            "### Problem Description\n"
            f"{problem}\n\n"
            "### Buggy Code\n"
            f"{code}\n\n"
            "Please follow the 3-stage internal process, and output only the final JSON object."
        )
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        prompt = self.lp_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True 
        )

        input_ids = self.lp_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).input_ids.to(self.lp_model.device)
        
        # 生成
        with torch.no_grad():
            output_ids = self.lp_model.generate(
                input_ids,
                generation_config=generation_config
            )

        response_text = self.lp_tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        
        # 后处理
        analysis_result = extract_json(response_text)
        
        if analysis_result is None:
            return {"bug_span": [[1, 1000]], "bug_summary": "parse_error", "planner_text": "Analyze the problem and the buggy code carefully, then provide the corrected code."}
        
        return analysis_result


    # ---------------- Fixer ----------------
    def run_fixer(self, problem_text, code, error, plan):
        """ Stage 2: Fixer uses the Plan as 'Teacher Instructions' to rewrite the code. """
        
        messages = [
            {"role": "system", "content": "You are an expert programmer. Your task is to fix the buggy code provided below, following the specific instructions given by the teacher."},
            {"role": "user", "content": f"""PROBLEM DESCRIPTION:\n{problem_text}\n\nBUGGY CODE:\n{code}\n\nERROR MESSAGE:\n{error}\n\nTEACHER INSTRUCTIONS (PLAN):
{plan}

Based on the instructions above, please output the complete, corrected Python code. Wrap the code in ```python ... ``` blocks."""}
        ]
        
        text = self.fixer_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.fixer_tokenizer(text, return_tensors="pt").to(self.fixer_model.device)
        
        with torch.no_grad():
            outputs = self.fixer_model.generate(
                **inputs, max_new_tokens=1024, temperature=0.2, do_sample=True 
            )
        
        response = self.fixer_tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return extract_code_from_fixer(response)

# ========================================
# 🔄 Main Loop (E2E Inference)
# ========================================
def main():
    parser = argparse.ArgumentParser()
    # 🚨 input_file 现在使用顶部的 DEFAULT_INPUT_FILE 作为默认值
    parser.add_argument("--input_file", type=str, default=DEFAULT_INPUT_FILE, help="Path to evaluation results json")
    # 🚨 output_file 现在使用顶部的 DEFAULT_OUTPUT_FILE 作为默认值
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Path to save the generated output file for evaluation.")
    args = parser.parse_args()

    # 1. Load Input Data
    print(f"Loading results from: {args.input_file}")
    with open(args.input_file, "r") as f:
        results = json.load(f)

    # 2. Init Pipeline
    pipeline = RepairPipeline()
    
    repaired_count = 0
    skipped_count = 0
    SAVE_INTERVAL = 10 

    print(f"Processing {len(results)} samples...")
    
    for i, entry in tqdm(enumerate(results), total=len(results)):
        
        q_id = entry.get("question_id", f"unknown_{i}")
        
        # 3. --- 跳过已修复的数据 ---
        if entry.get("pass@1") == 1.0:
            skipped_count += 1
            continue

        # 4. --- 准备输入数据 ---
        question_text = entry.get("question_content", entry.get("question_title", ""))
        code = entry.get("code_list", [""])[0]
        error_msg = parse_metadata_error(entry.get("metadata"))
        
        if not code or not code.strip(): continue

        # 5. --- EXECUTE E2E PIPELINE ---
        try:
            # L/P 模型生成 (使用 SFT Prompt)
            e2e_result = pipeline.run_e2e_analysis(question_text, code, error_msg)
            span = e2e_result["bug_span"]
            plan = e2e_result["planner_text"]
            summary = e2e_result["bug_summary"]

            # FIXER
            fixed_code = pipeline.run_fixer(question_text, code, error_msg, plan)
            
            # 6. --- UPDATE ENTRY ---
            entry["code_list"] = [fixed_code]
            entry["repair_info"] = {
                "original_error": error_msg,
                "bug_span": str(span),
                "bug_summary": summary,
                "teacher_plan": plan
            }
            
            # 7. 重置 metrics
            entry["pass@1"] = 0.0
            entry["graded_list"] = [] 
            entry["metadata"] = [] 
            
            repaired_count += 1
            
        except Exception as e:
            print(f"[{q_id}] Pipeline error: {e}")
            
        # 8. Periodic Save
        if (i + 1) % SAVE_INTERVAL == 0:
            with open(args.output_file, "w") as f:
                json.dump(results, f, indent=2)

    # Final Save
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*40)
    print(f"E2E Inference (SFT Prompt) process complete. Saved to {args.output_file}")
    print(f"Repaired: {repaired_count}")
    print(f"Skipped (Already Correct): {skipped_count}")
    print("="*40)

if __name__ == "__main__":
    main()