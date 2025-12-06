import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any

# 确保这里的路径与您 SFT 模型保存的 output_dir 一致
MODEL_PATH = "/home/ubuntu/finetune_e2e_new/model" 
TEST_FILE_PATH = "/home/ubuntu/finetune_e2e_new/data/test_intrinsic.json"
OUTPUT_FILE_PATH = "/home/ubuntu/finetune_e2e_new/data/preds_intrinsic.json"

# -----------------------------------------------------------------
# 1. JSON 提取器
# -----------------------------------------------------------------
def extract_json(raw: str) -> Dict[str, Any] | None:
    """Uses regex to find and parse a JSON object in the raw output."""
    try:
        # 查找第一个 '{' 和最后一个 '}'，提取中间的 JSON 字符串
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            return None
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    except Exception:
        return None


# -----------------------------------------------------------------
# 2. 推理生成函数 (与训练模板一致)
# -----------------------------------------------------------------
def generate_analysis(buggy_code: str, problem_content: str, model, tokenizer, max_len: int) -> Dict[str, Any]:
    """
    使用 SFT 后的模型生成 bug_span, bug_summary 和 planner_text 的 JSON 分析结果。
    """
    
   # --- 1. SYSTEM PROMPT (3 阶段指令) ---
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

    user_msg = (
        "### 问题描述\n"
        f"{problem}\n\n"
        "### 错误代码\n"
        f"{code}\n\n"  # <-- 将 'buggy' 替换为 'code'
        "请遵循 3 阶段内部流程，并只输出最终的 JSON 对象。"
    )
    
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    # 应用 Chat 模板并生成 prompt (add_generation_prompt=True)
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True 
    )

    # 推理配置
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(model.device)
    
    # 设置生成参数：强制使用贪婪搜索，控制 JSON 结束
    generation_config = model.generation_config
    generation_config.max_new_tokens = 2048
    # 1. 启用采样
    generation_config.do_sample = True 
    # 2. 调整温度
    generation_config.temperature = 0.8 
    # 3. 启用 Top-p 采样
    generation_config.top_p = 0.9 
    # 4. 启用 Top-k 采样
    generation_config.top_k = 50  
    
    # 确保在 } 结束或标准 EOS 结束
    generation_config.eos_token_id = [
        tokenizer.eos_token_id,              
        tokenizer.convert_tokens_to_ids('}') 
    ] 
    
    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            generation_config=generation_config
        )

    # 解码 (只取生成部分)
    response_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    
    # 后处理
    analysis_result = extract_json(response_text)
    
    if analysis_result is None:
        # 如果解析失败，返回包含默认值的字典
        return {"bug_span": [], "bug_summary": "parse_error", "planner_text": "parse_error", "raw_output": response_text}
    
    return analysis_result


# -----------------------------------------------------------------
# 3. 主程序：批量处理
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    # 使用训练配置中的 max_length
    MAX_SEQ_LENGTH = 4096 
    
    # 1. 加载模型和 tokenizer
    try:
        print(f"Loading model from {MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        exit()

    # 2. 加载测试数据
    try:
        with open(TEST_FILE_PATH, 'r') as f:
            test_data = json.load(f)
        print(f"Loaded {len(test_data)} test samples from {TEST_FILE_PATH}")
    except Exception as e:
        print(f"Error loading test file: {e}")
        exit()

    results = []

    # 3. 批量推理
    for idx, example in enumerate(test_data):
        print(f"\n=== Processing sample {idx+1}/{len(test_data)} ===")
        
        problem = example["question_content"]
        code = example["code_list"][0]

        # 运行生成
        analysis_dict = generate_analysis(code, problem, model, tokenizer, MAX_SEQ_LENGTH)
        
        # 🚨 关键修复：扁平化结果，只保留 question_title 和预测的 key/value
        results.append({
            "question_title": example["question_title"],
            "bug_span": analysis_dict.get("bug_span", []),
            "bug_summary": analysis_dict.get("bug_summary", "parse_error"),
            "planner_text": analysis_dict.get("planner_text", "parse_error"),
        })

        print(f"  > Predicted bug_span: {results[-1]['bug_span']}")
        print(f"  > Predicted bug_summary: {results[-1]['bug_summary']}")
        print(f"  > Predicted planner_text: {results[-1]['planner_text']}")

    # 4. 保存结果
    with open(OUTPUT_FILE_PATH, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n--- Inference Complete. Results saved to {OUTPUT_FILE_PATH} ---")