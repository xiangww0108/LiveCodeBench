"""
Evaluation-only script for fine-tuned planner model
Runs inference and computes metrics without training
"""

import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
from tqdm import tqdm
import numpy as np
import gc

# Configuration
MODEL_PATH = "./planner-finetuned"  # Your saved model
OUTPUT_FILE = "evaluation_results_final.json"
PREDICTIONS_FILE = "predictions_final.json"
MAX_LENGTH = 2048


def format_planner_input(example):
    """Format the input according to the planner prompt template"""
    prompt = f"""You are a precise and concise bug fixer (planner only).

Given:
- Problem: {example['question_content'][:500]}...
- Buggy code:
{example['code_list'][0] if isinstance(example['code_list'], list) else example['code_list']}

- Metadata: {example['metadata'][0] if isinstance(example['metadata'], list) else example['metadata']}
- Bug spans: {example['bug_span']}
- Bug summary: {example['bug_summary']}

Your task: Produce a short sequence of steps (2-6) describing how to fix the bug. Include a suggested corrected code snippet.

Plan:"""
    
    return prompt


def main():
    print("="*50)
    print("Planner Evaluation Script")
    print("="*50 + "\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load model and tokenizer
    print("Loading fine-tuned model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded!\n")
    
    # Load test data
    print("Loading test data from HuggingFace...")
    dataset = load_dataset(
        "Intellegen4/Qwen3-TrainTest-data",
        data_files="test-pre.json",
        split="train"
    )
    test_data = dataset.to_list()
    print(f"Loaded {len(test_data)} test examples\n")
    
    # Generate predictions
    print("="*50)
    print("Generating predictions...")
    print("="*50 + "\n")
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for example in tqdm(test_data, desc="Generating"):
            input_text = format_planner_input(example)
            
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = generated_text[len(input_text):].strip()
            
            predictions.append(prediction)
            references.append(example['bug_summary'])
    
    print("\nGeneration complete!")
    
    # Free GPU memory before loading evaluation models
    print("\nFreeing GPU memory...")
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compute metrics
    print("\n" + "="*50)
    print("Computing Metrics...")
    print("="*50 + "\n")
    
    # BERTScore with smaller model
    print("Computing BERTScore (using base model)...")
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-base-mnli",  # Smaller model to avoid OOM
        batch_size=8  # Process in smaller batches
    )
    
    # ROUGE
    print("Computing ROUGE...")
    rouge = evaluate.load("rouge")
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references
    )
    
    # BLEU
    print("Computing BLEU...")
    bleu = evaluate.load("bleu")
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
    
    # Save results
    results_dict = {
        'bertscore': {
            'precision': float(np.mean(bert_results['precision'])),
            'recall': float(np.mean(bert_results['recall'])),
            'f1': float(np.mean(bert_results['f1']))
        },
        'rouge': {
            'rouge1': float(rouge_results['rouge1']),
            'rouge2': float(rouge_results['rouge2']),
            'rougeL': float(rouge_results['rougeL'])
        },
        'bleu': float(bleu_results['bleu'])
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved to {OUTPUT_FILE}")
    
    # Save predictions
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump({
            'predictions': predictions,
            'references': references
        }, f, indent=2)
    print(f"Predictions saved to {PREDICTIONS_FILE}")
    
    # Print sample predictions
    print("\n" + "="*50)
    print("Sample Predictions (first 3)")
    print("="*50)
    for i in range(min(3, len(predictions))):
        print(f"\n--- Example {i+1} ---")
        print(f"Prediction: {predictions[i][:300]}...")
        print(f"\nReference: {references[i]}")
        print("-" * 50)
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
