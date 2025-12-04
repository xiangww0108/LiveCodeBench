"""
Fine-tune Qwen3-4B for planner task
Evaluates with BERTScore, ROUGE, and BLEU metrics
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import evaluate
from tqdm import tqdm
import numpy as np
from huggingface_hub import login

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # Qwen2.5-Coder-1.5B-Instruct model
OUTPUT_DIR = "./planner-finetuned-1.5b-temp0-max800"  # New output directory
HF_REPO = "Intellegen4/modular-planner-qwen2.5-coder-1.5b-temp0-max800"  # New repo for uploading

# Training hyperparameters
BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
MAX_LENGTH = 2048
WARMUP_RATIO = 0.05


def format_planner_input(example):
    """
    Format the input according to the planner prompt template
    """
    prompt = f"""You are a precise and concise bug fixer (planner only).

Given:
- Problem: {example['question_content'][:500]}...  # Truncate if too long
- Buggy code:
{example['code_list'][0] if isinstance(example['code_list'], list) else example['code_list']}

- Metadata: {example['metadata'][0] if isinstance(example['metadata'], list) else example['metadata']}
- Bug spans: {example['bug_span']}
- Bug summary: {example['bug_summary']}

Your task: Produce a short sequence of steps (2-6) describing how to fix the bug. Include a suggested corrected code snippet.

Plan:"""
    
    return prompt


def prepare_train_dataset():
    """Load and prepare training dataset from HuggingFace"""
    from datasets import load_dataset
    
    print("Loading training data from HuggingFace...")
    dataset = load_dataset(
        "Intellegen4/Qwen3-TrainTest-data",
        data_dir="modular-step-by-step",
        data_files="train-planner.json",
        split="train"
    )
    
    data = dataset.to_list()
    print(f"Loaded {len(data)} training examples")
    
    # Format data for training
    formatted_data = []
    for example in data:
        formatted_data.append({
            'input': format_planner_input(example),
            'output': example['planner_text']
        })
    
    return Dataset.from_list(formatted_data)


def prepare_test_dataset():
    """Load and prepare test dataset from HuggingFace"""
    from datasets import load_dataset
    
    print("Loading test data from HuggingFace...")
    dataset = load_dataset(
        "Intellegen4/Qwen3-TrainTest-data",
        data_files="test-pre.json",
        split="train"
    )
    
    data = dataset.to_list()
    print(f"Loaded {len(data)} test examples")
    
    # Extract only essential fields for planner inference
    formatted_data = []
    for example in data:
        formatted_data.append({
            'question_content': example['question_content'],
            'code_list': example['code_list'],
            'metadata': example['metadata'],
            'bug_span': example['bug_span'],
            'bug_summary': example['bug_summary'],
            'input': format_planner_input(example)
        })
    
    return formatted_data


def tokenize_function(examples, tokenizer):
    """Tokenize inputs and outputs"""
    # Combine input and output for training
    full_texts = [
        f"{inp}\n{out}" 
        for inp, out in zip(examples['input'], examples['output'])
    ]
    
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    
    # Also tokenize just inputs to know where to start generating from
    input_tokenized = tokenizer(
        examples['input'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    
    # Create labels (mask input part, only train on output)
    labels = []
    for i in range(len(tokenized['input_ids'])):
        input_len = len(input_tokenized['input_ids'][i])
        label = [-100] * input_len + tokenized['input_ids'][i][input_len:]
        labels.append(label)
    
    tokenized['labels'] = labels
    
    return tokenized


def evaluate_planner(model, tokenizer, test_data, device):
    """
    Evaluate the planner model using multiple metrics:
    - BERTScore
    - ROUGE
    - BLEU
    """
    print("\n" + "="*50)
    print("Starting evaluation...")
    print("="*50 + "\n")
    
    # Load metrics
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    predictions = []
    references = []
    
    model.eval()
    
    with torch.no_grad():
        for example in tqdm(test_data, desc="Generating predictions"):
            # Tokenize input
            inputs = tokenizer(
                example['input'],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated part (after the prompt)
            prediction = generated_text[len(example['input']):].strip()
            
            predictions.append(prediction)
            # Note: test-pre doesn't have planner_text, so we'll use bug_summary as proxy reference
            # In real evaluation, you'd need ground truth planner_text
            references.append(example['bug_summary'])
    
    # Compute BERTScore
    print("\nComputing BERTScore...")
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-base-mnli"  # Using base model instead of xlarge to save memory
    )
    
    # Compute ROUGE
    print("Computing ROUGE...")
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references
    )
    
    # Compute BLEU
    print("Computing BLEU...")
    # BLEU expects references as list of lists
    bleu_results = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"\nBERTScore:")
    print(f"  Precision: {np.mean(bert_results['precision']):.4f}")
    print(f"  Recall:    {np.mean(bert_results['recall']):.4f}")
    print(f"  F1:        {np.mean(bert_results['f1']):.4f}")
    
    print(f"\nROUGE:")
    print(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")
    
    print(f"\nBLEU: {bleu_results['bleu']:.4f}")
    print("="*50 + "\n")
    
    # Save sample predictions
    print("\nSample predictions (first 3):")
    for i in range(min(3, len(predictions))):
        print(f"\n--- Example {i+1} ---")
        print(f"Input (truncated): {test_data[i]['input'][:200]}...")
        print(f"\nPrediction: {predictions[i][:300]}...")
        print(f"\nReference (bug_summary): {references[i]}")
        print("-" * 50)
    
    return {
        'bertscore': {
            'precision': np.mean(bert_results['precision']),
            'recall': np.mean(bert_results['recall']),
            'f1': np.mean(bert_results['f1'])
        },
        'rouge': rouge_results,
        'bleu': bleu_results['bleu'],
        'predictions': predictions,
        'references': references
    }


def main():
    print("="*50)
    print("Planner Fine-tuning Script")
    print("="*50 + "\n")
    
    # Check CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.gradient_checkpointing_enable()
    
    # Print model info
    print(f"Model loaded: {MODEL_NAME}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    
    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset = prepare_train_dataset()
    
    # Tokenize
    print("Tokenizing training data...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=device == "cuda",
        report_to="none",
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        gradient_checkpointing=True,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator
    )
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate on test set
    print("\nLoading test data for evaluation...")
    test_data = prepare_test_dataset()
    
    results = evaluate_planner(model, tokenizer, test_data, device)
    
    # Save results
    results_path = f"{OUTPUT_DIR}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            'bertscore': results['bertscore'],
            'rouge': results['rouge'],
            'bleu': results['bleu']
        }, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    # Save predictions
    predictions_path = f"{OUTPUT_DIR}/predictions.json"
    with open(predictions_path, 'w') as f:
        json.dump({
            'predictions': results['predictions'],
            'references': results['references']
        }, f, indent=2)
    
    print(f"Predictions saved to {predictions_path}")
    
    print("\n" + "="*50)
    print("Training and evaluation complete!")
    print("="*50)


if __name__ == "__main__":
    main()