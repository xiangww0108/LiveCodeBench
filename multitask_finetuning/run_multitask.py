import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers import DataCollatorForSeq2Seq

from dataset import MultiTaskDataset
from utils import load_json

def main():
    config = load_json("multitask_finetuning/config.json")

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    train_ds = MultiTaskDataset(
        config["train_file"],
        tokenizer,
        max_length=config["max_seq_len"]
    )

    args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["grad_accum"],
        learning_rate=config["lr"],
        num_train_epochs=config["epochs"],
        warmup_ratio=config["warmup_ratio"],
        logging_steps=10,
        save_steps=200,
        bf16=True,
        report_to="none",
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(config["output_dir"])

if __name__ == "__main__":
    main()
