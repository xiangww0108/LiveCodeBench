import torch
import json
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)

# ================= CONFIGURATION =================
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
# Output directory for the saved model
OUTPUT_DIR = "./localizer_3B_FullSFT"
# Your Hugging Face Repo to push to
HUB_MODEL_ID = "Intellegen4/Qwen2.5-Coder-3B-Localizer-FullSFT"

# Hardware: g6e.4xlarge (48GB VRAM) settings
# 8192 context length fits comfortably in 48GB with a 3B model
MAX_SEQ_LENGTH = 8192  

# ================= DATA PROCESSING =================
def format_localizer_data(example):
    """
    Transforms raw JSON data into the Qwen Chat format.
    Adds line numbers to code to allow specific localization.
    """
    try:
        # 1. Add Line Numbers to Code
        raw_code = example['code_list'][0]
        code_lines = raw_code.split('\n')
        numbered_code = "\n".join([f"{i+1} | {line}" for i, line in enumerate(code_lines)])
        
        # 2. Extract Error Message
        # The metadata is a list containing a JSON string
        meta_str = example['metadata'][0]
        meta_json = json.loads(meta_str)
        error = meta_json.get('error_message', 'Runtime Error')

        # 3. Construct Chat Messages
        return {
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a code debugger. Locate the bug. Output ONLY a JSON list of line ranges like [[start, end]]."
                },
                {
                    "role": "user", 
                    "content": f"PROBLEM:\n{example['question_content']}\n\nERROR:\n{error}\n\nCODE:\n{numbered_code}"
                },
                {
                    "role": "assistant", 
                    "content": str(example['bug_span'])
                }
            ]
        }
    except Exception as e:
        return {"messages": []}

def train():
    print("--- 1. Loading Datasets ---")
    
    # A. Load Training Data (from modular-step-by-step folder)
    train_ds = load_dataset(
        "Intellegen4/Qwen3-TrainTest-data", 
        data_dir="modular-step-by-step", 
        data_files="train-localizer.json",
        split="train"
    )
    
    # B. Load Eval Data (from root test-pre.json because it has labels)
    # We cannot use test.json because it misses 'bug_span'
    try:
        eval_ds = load_dataset(
            "Intellegen4/Qwen3-TrainTest-data", 
            data_files="test-pre.json",
            split="train"
        )
    except Exception as e:
        print(f"Warning: Could not load test-pre.json ({e}). Splitting train set instead.")
        # Fallback: Split train set 90/10 if test-pre.json fails
        split = train_ds.train_test_split(test_size=0.1)
        train_ds = split['train']
        eval_ds = split['test']

    print(f"Train Size: {len(train_ds)}")
    print(f"Eval Size:  {len(eval_ds)}")

    # Apply Formatting
    train_dataset = train_ds.map(format_localizer_data, remove_columns=train_ds.column_names)
    eval_dataset = eval_ds.map(format_localizer_data, remove_columns=eval_ds.column_names)
    
    # Filter empty rows
    train_dataset = train_dataset.filter(lambda x: len(x["messages"]) > 0)
    eval_dataset = eval_dataset.filter(lambda x: len(x["messages"]) > 0)

    print("--- 2. Loading Model & Tokenizer ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model in BFloat16 (Native for L40S)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,    
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    print("--- 3. Configuring Training Arguments ---")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        
        # Batch Size for 48GB VRAM (g6e.4xlarge)
        # 3B Model + 8k Context = ~20-25GB static. 
        # Batch size 4 fills the rest efficiently.
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        
        # Optimizer
        optim="adamw_torch",            
        learning_rate=2e-5,             
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        
        # Hardware optimization
        bf16=True,                      
        gradient_checkpointing=True,    
        
        # Logging & Saving
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="steps",
        eval_steps=50,
        
        # Hub Upload
        push_to_hub=True,
        hub_model_id=HUB_MODEL_ID,
        
        # Data params
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="messages",
        packing=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("--- 4. Starting Training ---")
    trainer.train()
    
    print("--- 5. Saving and Uploading ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    trainer.push_to_hub()
    print("Done!")

if __name__ == "__main__":
    train()
    