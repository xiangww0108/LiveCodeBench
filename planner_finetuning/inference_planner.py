"""
Inference script for fine-tuned planner model
Can use either local model or HuggingFace model
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm


def format_planner_input(example):
    """Format input for planner inference"""
    # Handle code_list (might be list or string)
    code = example['code_list'][0] if isinstance(example['code_list'], list) else example['code_list']
    
    # Handle metadata (might be list or string)
    metadata = example['metadata'][0] if isinstance(example['metadata'], list) else example['metadata']
    
    # Truncate question if too long
    question = example['question_content']
    if len(question) > 500:
        question = question[:500] + "..."
    
    prompt = f"""You are a precise and concise bug fixer (planner only).

Given:
- Problem: {question}
- Buggy code:
{code}

- Metadata: {metadata}
- Bug spans: {example['bug_span']}
- Bug summary: {example['bug_summary']}

Your task: Produce a short sequence of steps (2-6) describing how to fix the bug. Include a suggested corrected code snippet.

Plan:"""
    
    return prompt


def load_model(model_path, device="cuda"):
    """
    Load the fine-tuned model
    
    Args:
        model_path: Path to model (local or HF repo)
        device: Device to load on
    """
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.eval()
    print("Model loaded successfully!\n")
    
    return model, tokenizer


def generate_plan(model, tokenizer, input_text, device="cuda", max_new_tokens=512):
    """Generate a repair plan for given input"""
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    plan = generated_text[len(input_text):].strip()
    
    return plan


def run_inference(model_path, test_file, output_file, device="cuda"):
    """
    Run inference on test file and save results
    
    Args:
        model_path: Path to fine-tuned model
        test_file: Path to test-pre-with-plan.json
        output_file: Path to save predictions
        device: Device to use
    """
    # Load model
    model, tokenizer = load_model(model_path, device)
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test examples\n")
    
    # Generate plans
    results = []
    print("Generating repair plans...")
    
    for example in tqdm(test_data):
        # Format input
        input_text = format_planner_input(example)
        
        # Generate plan
        plan = generate_plan(model, tokenizer, input_text, device)
        
        # Store result
        results.append({
            'question_title': example.get('question_title', ''),
            'bug_summary': example['bug_summary'],
            'bug_span': example['bug_span'],
            'planner_text': example.get('planner_text', ''),  # Ground truth
            'generated_plan': plan,
            'input_prompt': input_text  # For debugging
        })
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Inference complete! Results saved to {output_file}")
    
    # Print sample outputs
    print("\n" + "="*70)
    print("SAMPLE OUTPUTS (first 2 examples)")
    print("="*70)
    
    for i, result in enumerate(results[:2]):
        print(f"\n--- Example {i+1}: {result['question_title']} ---")
        print(f"\nBug Summary: {result['bug_summary']}")
        print(f"\nBug Span: {result['bug_span']}")
        print(f"\nGenerated Plan:\n{result['generated_plan']}")
        print("-"*70)


def interactive_mode(model_path, device="cuda"):
    """Interactive mode for testing individual examples"""
    print("\n" + "="*70)
    print("INTERACTIVE PLANNER MODE")
    print("="*70)
    print("Enter example details (or 'quit' to exit)\n")
    
    model, tokenizer = load_model(model_path, device)
    
    while True:
        try:
            print("\n" + "-"*70)
            question = input("Problem description: ").strip()
            if question.lower() == 'quit':
                break
            
            code = input("Buggy code: ").strip()
            metadata = input("Metadata (error info): ").strip()
            bug_span = input("Bug span (e.g., [[5, 10]]): ").strip()
            bug_summary = input("Bug summary: ").strip()
            
            # Create example dict
            example = {
                'question_content': question,
                'code_list': [code],
                'metadata': [metadata],
                'bug_span': eval(bug_span),  # Convert string to list
                'bug_summary': bug_summary
            }
            
            # Format and generate
            input_text = format_planner_input(example)
            plan = generate_plan(model, tokenizer, input_text, device)
            
            print("\n" + "="*70)
            print("GENERATED PLAN:")
            print("="*70)
            print(plan)
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planner inference")
    parser.add_argument("--model_path", type=str, default="./planner-finetuned",
                        help="Path to fine-tuned model (local or HF repo)")
    parser.add_argument("--test_file", type=str, default="test-pre-with-plan.json",
                        help="Path to test-pre-with-plan.json (contains planner_text ground truth)")
    parser.add_argument("--output_file", type=str, default="planner_predictions.json",
                        help="Output file for predictions")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode(args.model_path, args.device)
    else:
        run_inference(
            args.model_path,
            args.test_file,
            args.output_file,
            args.device
        )