import json
import argparse
import multiprocessing
import os
import sys
from tqdm import tqdm
from datasets import load_dataset
from dataclasses import fields
from datetime import datetime
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem

# Import the standard execution function
try:
    from lcb_runner.evaluation.utils_execute import check_correctness
except ImportError:
    print("Error: Could not import lcb_runner. Make sure you are in the LiveCodeBench root directory.")
    sys.exit(1)

def safe_load_problems():
    """
    Loads the dataset safely, handling potential format differences 
    between the Parquet version and the script version.
    """
    print("Loading problems from Hugging Face...")
    try:
        # Try loading with trust_remote_code=True first
        dataset = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"Standard load failed ({e}). Loading generic parquet version...")
        dataset = load_dataset("livecodebench/code_generation", split="test")

    print(f"Loaded {len(dataset)} raw problems. Converting to CodeGenerationProblem objects...")
    
    # Identify valid fields for the class to prevent "unexpected keyword argument" errors
    try:
        valid_keys = {f.name for f in fields(CodeGenerationProblem)}
    except TypeError:
        # Fallback if it's not a standard dataclass (though it usually is)
        valid_keys = None

    problem_map = {}
    first_error = None
    
    for item in dataset:
        try:
            # 1. Filter out extra columns (like __index_level_0__)
            if valid_keys:
                filtered_item = {k: v for k, v in item.items() if k in valid_keys}
            else:
                filtered_item = item.copy()

            # 2. SANITIZATION: Convert Parquet types to what CodeGenerationProblem expects
            
            # Handle Date: If it's a datetime object, convert to string so __post_init__ can parse it
            if "contest_date" in filtered_item:
                val = filtered_item["contest_date"]
                if not isinstance(val, str):
                    if hasattr(val, "isoformat"):
                        filtered_item["contest_date"] = val.isoformat()
                    else:
                        filtered_item["contest_date"] = str(val)

            # Handle Test Cases: If they are lists, dump to JSON string
            for key in ["public_test_cases", "private_test_cases"]:
                if key in filtered_item:
                    val = filtered_item[key]
                    if isinstance(val, list):
                        filtered_item[key] = json.dumps(val)
                    elif val is None:
                        # Some parquet rows might have None
                        filtered_item[key] = "[]"

            # Handle Metadata: If dict, dump to JSON string
            if "metadata" in filtered_item:
                val = filtered_item["metadata"]
                if isinstance(val, dict):
                    filtered_item["metadata"] = json.dumps(val)

            # Create the problem object
            prob = CodeGenerationProblem(**filtered_item)
            problem_map[prob.question_id] = prob
            
        except Exception as e:
            if first_error is None:
                first_error = e
                print(f"\n[DEBUG] First conversion error: {type(e).__name__}: {e}")
                print(f"[DEBUG] Problem keys: {list(item.keys())}")
            continue
            
    if not problem_map:
        print("\nCRITICAL: Failed to convert all items.")
        if first_error:
            print("Please check the '[DEBUG]' message above for details.")
    else:
        print(f"Successfully converted {len(problem_map)} problems.")

    return problem_map

def evaluate_sample(args):
    """
    Worker function to run a single test case.
    """
    problem, code, timeout = args
    try:
        # Attempt execution
        result = check_correctness(problem, code, timeout=timeout)
        return result
    except Exception as e:
        return {"passed": False, "error": f"Runner Exception: {str(e)}"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to your repaired results json")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save evaluation results")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--timeout", type=float, default=6.0, help="Execution timeout per sample")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.replace(".json", "_eval.json")

    # 1. Load Problems
    problem_map = safe_load_problems()
    if not problem_map:
        print("Error: No problems loaded. Exiting.")
        return

    # 2. Load Predictions
    print(f"Loading predictions from {args.input_file}...")
    try:
        with open(args.input_file, "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found {args.input_file}")
        return

    # 3. Prepare Tasks
    eval_tasks = []
    task_indices = [] # Map task_id -> (result_index, code_index)
    
    print("Preparing evaluation tasks...")
    for i, res in enumerate(results):
        q_id = res.get("question_id")
        if not q_id or q_id not in problem_map:
            continue
            
        prob = problem_map[q_id]
        
        # We evaluate every code snippet in the 'code_list'
        code_list = res.get("code_list", [])
        if isinstance(code_list, str):
            code_list = [code_list]
            
        for j, code in enumerate(code_list):
            eval_tasks.append((prob, code, args.timeout))
            task_indices.append((i, j))

    if not eval_tasks:
        print("No tasks to evaluate. Check if 'question_id' in input matches dataset IDs.")
        return

    print(f"Evaluating {len(eval_tasks)} samples with {args.num_workers} workers...")
    
    # 4. Run Execution (Parallel)
    if args.num_workers > 1:
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            eval_results = list(tqdm(pool.imap(evaluate_sample, eval_tasks), total=len(eval_tasks)))
    else:
        # Sequential fallback
        eval_results = []
        for task in tqdm(eval_tasks):
            eval_results.append(evaluate_sample(task))

    # 5. Process Results
    # Initialize graded_lists if missing
    for res in results:
        if "graded_list" not in res:
            res["graded_list"] = []

    pass_count = 0
    total_count = 0
    
    # DEBUG: Track failures to print later
    sample_failures = []

    # Reset graded_list for items we actually evaluated
    processed_indices = set(idx for idx, _ in task_indices)
    for idx in processed_indices:
        results[idx]["graded_list"] = []

    for (res_idx, code_idx), status in zip(task_indices, eval_results):
        # Interpret result
        is_passed = False
        error_msg = None
        
        if isinstance(status, bool):
            is_passed = status
        elif isinstance(status, dict):
            is_passed = status.get("passed", False)
            # Capture error if present and failed
            if not is_passed:
                # Common LCB return structure might be complex, try to extract error info
                error_msg = status.get("result", "Unknown Failure")
                if "error" in status:
                     error_msg = status["error"]
        
        # Append to the correct result entry
        results[res_idx]["graded_list"].append(is_passed)
        
        if is_passed:
            pass_count += 1
        else:
            if len(sample_failures) < 5: # Keep first 5 failures for debug
                sample_failures.append({
                    "id": results[res_idx].get("question_id"),
                    "status": status,
                    "error": error_msg
                })

        total_count += 1

    # 6. Compute Final Stats
    print("\nComputing statistics...")
    for res in results:
        graded = res.get("graded_list", [])
        if graded:
            res["pass@1"] = sum(graded) / len(graded)
        else:
            res["pass@1"] = 0.0

    print(f"Total Evaluated: {total_count}")
    print(f"Total Passed:    {pass_count}")
    print(f"Overall Pass@1:  {pass_count/total_count:.2%}" if total_count > 0 else "Overall Pass@1: 0%")

    if pass_count == 0 and total_count > 0:
        print("\n[DEBUG] ALL TESTS FAILED. Inspecting first few errors:")
        for fail in sample_failures:
            print(f"ID: {fail['id']} | Error: {fail['error']}")
            print(f"Raw Status: {fail['status']}\n")

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to {args.output_file}")

if __name__ == "__main__":
    main()
    