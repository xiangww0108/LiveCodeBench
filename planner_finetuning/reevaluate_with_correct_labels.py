"""
Re-evaluate existing predictions using correct ground truth (planner_text)
Loads predictions.json and test-pre-with-plan.json, then computes metrics
"""

import json
import torch
from datasets import load_dataset
import evaluate
import numpy as np
import argparse
from tqdm import tqdm


def load_predictions_file(predictions_file):
    """Load predictions from JSON file"""
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'predictions' in data:
        predictions = data['predictions']
    else:
        raise ValueError(f"Expected dict with 'predictions' key, got {type(data)}")
    
    print(f"✓ Loaded {len(predictions)} predictions")
    return predictions


def load_ground_truth():
    """Load ground truth from HuggingFace dataset"""
    print("Loading ground truth from HuggingFace (test-pre-with-plan.json)...")
    dataset = load_dataset(
        "Intellegen4/Qwen3-TrainTest-data",
        data_files="test-pre-with-plan.json",
        split="train"
    )
    references = [example['planner_text'] for example in dataset]
    print(f"✓ Loaded {len(references)} ground truth labels")
    return references


def evaluate_predictions(predictions, references):
    """Compute evaluation metrics"""
    assert len(predictions) == len(references), \
        f"Mismatch: {len(predictions)} predictions vs {len(references)} references"
    
    print("\n" + "="*70)
    print("Computing Metrics...")
    print("="*70 + "\n")
    
    # BERTScore with smaller model
    print("Computing BERTScore (using base model)...")
    bertscore = evaluate.load("bertscore")
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-base-mnli",
        batch_size=8
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
    
    return bert_results, rouge_results, bleu_results


def print_results(bert_results, rouge_results, bleu_results):
    """Print evaluation results"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS (Using Correct Ground Truth: planner_text)")
    print("="*70)
    print(f"\nBERTScore:")
    print(f"  Precision: {np.mean(bert_results['precision']):.4f}")
    print(f"  Recall:    {np.mean(bert_results['recall']):.4f}")
    print(f"  F1:        {np.mean(bert_results['f1']):.4f}")
    
    print(f"\nROUGE:")
    print(f"  ROUGE-1: {rouge_results['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_results['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_results['rougeL']:.4f}")
    
    print(f"\nBLEU: {bleu_results['bleu']:.4f}")
    print("="*70 + "\n")


def save_results(bert_results, rouge_results, bleu_results, output_file):
    """Save results to JSON"""
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
        'bleu': float(bleu_results['bleu']),
        'note': 'Evaluated using planner_text as ground truth from test-pre-with-plan.json'
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate predictions using correct ground truth (planner_text)"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for evaluation results (default: evaluation_results_corrected.json in same dir)"
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        import os
        pred_dir = os.path.dirname(args.predictions)
        args.output = os.path.join(pred_dir, "evaluation_results_corrected.json")
    
    print("="*70)
    print("RE-EVALUATING WITH CORRECT GROUND TRUTH")
    print("="*70)
    print(f"\nPredictions: {args.predictions}")
    print(f"Output:      {args.output}")
    print(f"Ground truth: planner_text from test-pre-with-plan.json\n")
    
    # Load data
    predictions = load_predictions_file(args.predictions)
    references = load_ground_truth()
    
    # Evaluate
    bert_results, rouge_results, bleu_results = evaluate_predictions(predictions, references)
    
    # Print and save
    print_results(bert_results, rouge_results, bleu_results)
    save_results(bert_results, rouge_results, bleu_results, args.output)
    
    print("\n✅ Re-evaluation complete!")
    print(f"\nOld incorrect evaluation: Used bug_summary (localizer output)")
    print(f"New correct evaluation:   Uses planner_text (planner ground truth)")


if __name__ == "__main__":
    main()
