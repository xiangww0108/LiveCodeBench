"""
Standalone evaluation script for planner model
Evaluates generated plans against reference using BERTScore, ROUGE, BLEU
"""

import json
import evaluate
import numpy as np
from tqdm import tqdm
import argparse


def load_predictions(predictions_file):
    """Load predictions from JSON file"""
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, dict) and 'predictions' in data:
        predictions = data['predictions']
        references = data.get('references', data.get('bug_summary', []))
    elif isinstance(data, list):
        # Assume it's list of dicts with 'generated_plan' and 'bug_summary'
        predictions = [item.get('generated_plan', '') for item in data]
        references = [item.get('bug_summary', '') for item in data]
    else:
        raise ValueError("Unexpected format for predictions file")
    
    return predictions, references


def evaluate_predictions(predictions, references):
    """
    Evaluate predictions using multiple metrics
    
    Args:
        predictions: List of generated plans
        references: List of reference texts (bug summaries or ground truth plans)
    """
    print(f"\nEvaluating {len(predictions)} predictions...")
    print("="*70)
    
    # Load metrics
    print("Loading evaluation metrics...")
    bertscore = evaluate.load("bertscore")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    
    # Compute BERTScore
    print("\nComputing BERTScore (this may take a few minutes)...")
    bert_results = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-xlarge-mnli",
        verbose=True
    )
    
    # Compute ROUGE
    print("\nComputing ROUGE scores...")
    rouge_results = rouge.compute(
        predictions=predictions,
        references=references
    )
    
    # Compute BLEU
    print("\nComputing BLEU score...")
    bleu_results = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )
    
    # Calculate statistics
    results = {
        'bertscore': {
            'precision': {
                'mean': float(np.mean(bert_results['precision'])),
                'std': float(np.std(bert_results['precision'])),
                'min': float(np.min(bert_results['precision'])),
                'max': float(np.max(bert_results['precision']))
            },
            'recall': {
                'mean': float(np.mean(bert_results['recall'])),
                'std': float(np.std(bert_results['recall'])),
                'min': float(np.min(bert_results['recall'])),
                'max': float(np.max(bert_results['recall']))
            },
            'f1': {
                'mean': float(np.mean(bert_results['f1'])),
                'std': float(np.std(bert_results['f1'])),
                'min': float(np.min(bert_results['f1'])),
                'max': float(np.max(bert_results['f1']))
            }
        },
        'rouge': {
            'rouge1': float(rouge_results['rouge1']),
            'rouge2': float(rouge_results['rouge2']),
            'rougeL': float(rouge_results['rougeL'])
        },
        'bleu': float(bleu_results['bleu'])
    }
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print("\n📊 BERTScore (semantic similarity):")
    print(f"  Precision: {results['bertscore']['precision']['mean']:.4f} ± {results['bertscore']['precision']['std']:.4f}")
    print(f"  Recall:    {results['bertscore']['recall']['mean']:.4f} ± {results['bertscore']['recall']['std']:.4f}")
    print(f"  F1:        {results['bertscore']['f1']['mean']:.4f} ± {results['bertscore']['f1']['std']:.4f}")
    
    print(f"\n📊 ROUGE (n-gram overlap):")
    print(f"  ROUGE-1: {results['rouge']['rouge1']:.4f}")
    print(f"  ROUGE-2: {results['rouge']['rouge2']:.4f}")
    print(f"  ROUGE-L: {results['rouge']['rougeL']:.4f}")
    
    print(f"\n📊 BLEU (overall quality): {results['bleu']:.4f}")
    
    print("\n" + "="*70)
    
    # Analyze distribution
    print("\n📈 BERTScore F1 Distribution:")
    f1_scores = bert_results['f1']
    ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
    for low, high in ranges:
        count = sum(1 for score in f1_scores if low <= score < high)
        pct = count / len(f1_scores) * 100
        print(f"  [{low:.2f}, {high:.2f}): {count:3d} examples ({pct:5.1f}%)")
    
    return results, bert_results


def analyze_worst_cases(predictions, references, bert_results, n=5):
    """Analyze the worst performing predictions"""
    print("\n" + "="*70)
    print(f"WORST {n} PREDICTIONS (by BERTScore F1)")
    print("="*70)
    
    # Get indices of worst predictions
    f1_scores = bert_results['f1']
    worst_indices = np.argsort(f1_scores)[:n]
    
    for rank, idx in enumerate(worst_indices, 1):
        print(f"\n--- Rank {rank} (F1: {f1_scores[idx]:.4f}) ---")
        print(f"\nReference:\n{references[idx][:300]}...")
        print(f"\nPrediction:\n{predictions[idx][:300]}...")
        print("-"*70)


def analyze_best_cases(predictions, references, bert_results, n=5):
    """Analyze the best performing predictions"""
    print("\n" + "="*70)
    print(f"BEST {n} PREDICTIONS (by BERTScore F1)")
    print("="*70)
    
    # Get indices of best predictions
    f1_scores = bert_results['f1']
    best_indices = np.argsort(f1_scores)[-n:][::-1]
    
    for rank, idx in enumerate(best_indices, 1):
        print(f"\n--- Rank {rank} (F1: {f1_scores[idx]:.4f}) ---")
        print(f"\nReference:\n{references[idx][:300]}...")
        print(f"\nPrediction:\n{predictions[idx][:300]}...")
        print("-"*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate planner predictions")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions JSON file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                        help="Output file for evaluation results")
    parser.add_argument("--analyze", action="store_true",
                        help="Show best/worst case analysis")
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    predictions, references = load_predictions(args.predictions)
    
    print(f"✓ Loaded {len(predictions)} predictions")
    print(f"  Avg prediction length: {np.mean([len(p) for p in predictions]):.0f} chars")
    print(f"  Avg reference length: {np.mean([len(r) for r in references]):.0f} chars")
    
    # Evaluate
    results, bert_results = evaluate_predictions(predictions, references)
    
    # Save results
    print(f"\n💾 Saving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Detailed analysis
    if args.analyze:
        analyze_worst_cases(predictions, references, bert_results)
        analyze_best_cases(predictions, references, bert_results)
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()