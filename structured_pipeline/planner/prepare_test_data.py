"""
Prepare test data for planner inference
Extracts only essential fields from test-pre.json
"""

import json
import argparse


def prepare_test_data(input_file, output_file):
    """
    Extract only essential fields for planner inference
    
    Required fields:
    - question_content: Problem description
    - code_list: Buggy code
    - metadata: Error information
    - bug_span: From localizer (buggy line ranges)
    - bug_summary: From localizer (bug description)
    
    Optional (for analysis):
    - question_title: For identification
    """
    print(f"Loading test data from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # Extract essential fields
    prepared_data = []
    for example in data:
        prepared_example = {
            'question_title': example.get('question_title', ''),
            'question_content': example['question_content'],
            'code_list': example['code_list'],
            'metadata': example['metadata'],
            'bug_span': example['bug_span'],
            'bug_summary': example['bug_summary']
        }
        prepared_data.append(prepared_example)
    
    # Save prepared data
    print(f"Saving prepared data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(prepared_data, f, indent=2)
    
    print(f"✓ Prepared {len(prepared_data)} test examples")
    print(f"\nSample prepared example:")
    print(json.dumps(prepared_data[0], indent=2)[:500] + "...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare test data for planner")
    parser.add_argument("--input", type=str, default="test-pre.json",
                        help="Input test-pre.json file")
    parser.add_argument("--output", type=str, default="test-planner.json",
                        help="Output file with essential fields only")
    
    args = parser.parse_args()
    
    prepare_test_data(args.input, args.output)