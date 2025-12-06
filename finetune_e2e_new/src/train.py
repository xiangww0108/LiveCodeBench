import json
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig 
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================
#   Format E2E multistage sample → messages
#   【只保留 bug_span, bug_summary, planner_text】
# ============================================
def format_e2e(example):
    buggy = example["code_list"][0]
    problem = example["question_content"]

    span = example["bug_span"]
    summary = example["bug_summary"]
    planner = example["planner_text"]
    
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
        "### PROBLEM DESCRIPTION\n"
        f"{problem}\n\n"
        "### BUGGY CODE\n"
        f"{buggy}\n\n"
        "Follow the 3-stage internal process and output ONLY the final JSON object."
    )

    # --- 2. ASSISTANT MESSAGE (只保留 3 个字段) ---
    assistant_msg = json.dumps({
        "bug_span": span,
        "bug_summary": summary,
        "planner_text": planner,
    }, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = json.load(open(args.config))

    model_name = cfg["model_name"]
    max_len = int(cfg["max_length"]) # 从配置中读取 4096

    raw = load_dataset(
        "Intellegen4/Qwen3-TrainTest-data",
        data_files="end-to-end/train.json",
        split="train"
    )
    ds = raw.map(format_e2e, remove_columns=raw.column_names)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # 🚨 关键修复：删除手动 preprocess 函数和 ds.map(preprocess) 调用
    # SFTTrainer 会自动处理编码和正确的标签掩码。

    # 使用 SFTConfig 配置训练参数
    config = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"],
        label_smoothing_factor=cfg.get("label_smoothing", 0.0),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.0),
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        
        # 🚨 修复参数名：使用 max_seq_length 来设置 4096
        max_length=max_len, 
        
        dataset_text_field=None, 
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=ds, # 直接传入 messages 格式的数据集
    )

    print("Starting training…")
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()