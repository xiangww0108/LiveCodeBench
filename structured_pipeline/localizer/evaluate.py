import torch
import json
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the path to your saved model (or the Hub ID after uploading)
MODEL_PATH = "./localizer_3B_FullSFT" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_span(text):
    """
    Robustly extracts the FIRST JSON list of lists from text.
    Returns [] if not found.
    """
    try:
        # Regex to find [[...]] pattern
        match = re.search(r"\[\[.*?\]\]", text)
        if match:
            return json.loads(match.group(0))
    except:
        pass
    return []

def calculate_iou_recall(pred_spans, true_spans):
    """
    Calculates IoU and Recall based on set overlap of line numbers.
    """
    # 1. Convert Predicted Spans to a Set of Line Numbers
    pred_lines = set()
    if isinstance(pred_spans, list):
        for span in pred_spans:
            if isinstance(span, list) and len(span) == 2:
                # Add range (inclusive)
                # Ensure start <= end to avoid empty ranges
                start, end = min(span), max(span)
                pred_lines.update(range(start, end + 1))

    # 2. Convert True Spans to a Set of Line Numbers
    true_lines = set()
    if isinstance(true_spans, list):
        for span in true_spans:
            if isinstance(span, list) and len(span) == 2:
                start, end = min(span), max(span)
                true_lines.update(range(start, end + 1))
    
    if len(true_lines) == 0:
        return 0.0, 0.0 
        
    # 3. Calculate Intersection and Union
    intersection = len(pred_lines.intersection(true_lines))
    union = len(pred_lines.union(true_lines))
    
    # 4. Compute Metrics
    iou = intersection / union if union > 0 else 0.0
    recall = intersection / len(true_lines)
    
    return iou, recall

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.bfloat16, 
        device_map=DEVICE
    )
    
    print("Loading Test Dataset (test-pre.json)...")
    # We use test-pre.json because it contains the 'bug_span' labels
    test_ds = load_dataset(
        "Intellegen4/Qwen3-TrainTest-data", 
        data_files="test-pre.json", 
        split="train"
    )
    
    print(f"Evaluating on {len(test_ds)} examples...")
    
    ious = []
    recalls = []

    for i, example in enumerate(test_ds):
        # Prepare Input
        raw_code = example['code_list'][0]
        code_lines = raw_code.split('\n')
        numbered_code = "\n".join([f"{i+1} | {line}" for i, line in enumerate(code_lines)])
        
        meta_str = example['metadata'][0]
        meta_json = json.loads(meta_str)
        error = meta_json.get('error_message', 'Runtime Error')

        messages = [
            {"role": "system", "content": "You are a code debugger. Locate the bug. Output ONLY a JSON list of line ranges like [[start, end]]."},
            {"role": "user", "content": f"PROBLEM:\n{example['question_content']}\n\nERROR:\n{error}\n\nCODE:\n{numbered_code}"}
        ]
        
        # Generate Prediction
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=50, 
                temperature=0.1, 
                do_sample=False
            )
            
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Calculate Metrics
        pred_spans = parse_span(response)
        true_spans = example['bug_span']
        
        iou, recall = calculate_iou_recall(pred_spans, true_spans)
        ious.append(iou)
        recalls.append(recall)
        
        if i % 10 == 0:
            print(f"Ex {i}: IoU={iou:.2f} | Recall={recall:.2f} | Pred: {pred_spans} | True: {true_spans}")

    print("\n================ RESULTS ================")
    print(f"Mean IoU:    {np.mean(ious):.4f}")
    print(f"Mean Recall: {np.mean(recalls):.4f}")

if __name__ == "__main__":
    evaluate()
    