import json
import os

from anthropic import HUMAN_PROMPT, AI_PROMPT

from lcb_runner.lm_styles import LMStyle


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you are helping a user correct a error program for code competition. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the entire executable program. You must put the entire fixed executable program within code delimiters."

    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. You must put the entired fixed program within code delimiters only for once., for example: 
```python 
# YOUR CODE HERE
```"""

    FORMATTING_REPEAT = f"First reason about the code providing a textual explanation of what is wrong with the code and then generate a fixed of the program enclosed code delimiters."

    FORMATTING_MESSAGE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."


# def truncate_io(io):
#     if len(str(io)) > 200:
#         io = str(io)[:200] + "...."
#     return io


def get_check_prompt(question: str, result, metadata):
    ## assumes i/o examples are already truncated!
    ## less pressure on storing 10 MB json because on a single large input-output pair
    # result_by_test_case = result
    # assert len(metadata) == 1, f"metadata = {metadata}"
    # metadata = metadata[0]
    
    # Handle case when code passed (result is True)
    if result:
        return "The code passed all test cases."
    
    # Handle case when code failed (result is False)
    try:
        metadata = json.loads(metadata)
    except (json.JSONDecodeError, TypeError):
        return "The above code is incorrect and got a runtime error."
    
    if "error_code" not in metadata:
        return "The above code is incorrect."
        
    if metadata["error_code"] == -1:
        # compilation error
        message = f"The above code is incorrect and got the following compilation error.\n{metadata.get('error_message', 'Compilation error')}"
    elif metadata["error_code"] == -2:
        # wrong answer
        message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata.get('inputs', 'N/A')}\nGenerated Output: {metadata.get('output', 'N/A')}\nExpected: {metadata.get('expected', 'N/A')}\nError: {metadata.get('error_message', 'Wrong answer')}"
    elif metadata["error_code"] == -3:
        # runtime error
        message = f"The above code is incorrect and got a runtime error.\nInput: {metadata.get('inputs', 'N/A')}\nExpected: {metadata.get('expected', 'N/A')}\n{metadata.get('error_message', 'Runtime error')}"
    elif metadata["error_code"] == -4:
        # time limit exceeded / timeout
        message = f"The above code is incorrect and got time limit exceeded.\nInput: {metadata.get('inputs', 'N/A')}\nExpected: {metadata.get('expected', 'N/A')}\n{metadata.get('error_message', 'Time limit exceeded')}"
    else:
        message = f"The above code is incorrect (error code: {metadata['error_code']})."
        
    return message


def get_generic_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Answer:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_cllama_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question\n{question}\n\n"
    prompt += f"### Answer\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_deepseekcode_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"### Response:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_magicoder_question_template_answer(question: str, code, result, metadata):
    prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += f"@@ Response \n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_mixtral_question_template_answer(question: str, code, result, metadata):
    prompt = f"Question:\n"
    prompt += f"{question}\n\n"
    prompt += f"Answer:\n\n"
    prompt += f"```python\n\n{code}\n``\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_wizard_question_template_answer(question: str, code, result, metadata):
    prompt = f"""### Instruction: You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once., for example:
    ```python
    # YOUR CODE HERE
    ```
"""
    prompt += f"{question}\n\n"
    prompt += f"### Response:```python\n\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_qwen_cot_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Answer:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n"
    
    # Zero-shot CoT instructions 
    prompt += (
        "Think step by step about what is causing this failure and how to fix it, "
        "then produce a corrected version of the code.\n\n"
    )
    
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_qwen_structured_cot_question_template_answer(question: str, code, result, metadata):
    prompt = f"### Question:\n{question}\n\n"
    prompt += f"### Existing Code:\n```python\n{code}\n```\n\n"
    prompt += get_check_prompt(question, result, metadata) + "\n\n"

    # structured CoT debugging plan
    prompt += (
        "### Debugging Plan (think step by step)\n"
        "Please understand the requirement and the failure signal, and write a rough\n"
        "debugging plan in the following structure. Fill in the items in natural language.\n\n"
        "Input:\n"
        "  - <short description of the function inputs>\n"
        "Output:\n"
        "  - <short description of the expected output / behavior>\n"
        "Bug analysis:\n"
        "  1: <what the current code is doing>\n"
        "  2: <why this leads to the failing output or error for the given test case>\n"
        "Fix plan:\n"
        "  1: <the main code change(s) needed>\n"
        "  2: <any important edge cases or conditions to handle>\n\n"
        "After writing this plan, you will then provide the fixed code.\n\n"
    )
    # For structured-CoT we intentionally avoid including an example fenced block
    # here because some models may echo it, which confuses code extraction.
    prompt += "### Answer: Provide your '### Debugging Plan' followed by '### Fixed Code' with a single Python code block containing the entire corrected program. Do not include any other fenced code blocks.\n\n"
    return prompt

def get_phind_question_template_answer(question: str, code, result, metadata):
    prompt = f"{question}\n\n"
    prompt += f"```python\n{code}\n``` \n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Assistant"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt

def get_qwen_question_template_answer(question: str, code, result, metadata):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "abacusai/Dracarys-72B-Instruct", padding_side="left", use_fast=False
    )
    prompt = f"""### Instruction: You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entired fixed program within code delimiters only for once., for example:
    ```python
    # YOUR CODE HERE
    ```\n\n
"""
    prompt += f"Question:\n{question}\n\n"
    prompt += f"```python\n{code}\n``` \n\n"
    prompt += get_check_prompt(question, result, metadata)
    prompt += f"\n\n### Assistant"
    prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
    prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"

    messages = [
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt

def format_prompt_self_repair(
    question: str, LanguageModelStyle: LMStyle, code, result, metadata
) -> str:
    if result:
        # The code is accepted, no need to change anything.
        return ""
    if LanguageModelStyle == LMStyle.OpenAIChat:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                )
                + "\n\n"
                + PromptConstants.FORMATTING_REPEAT,
            },
        ]
        return chat_messages
    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    elif LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n\n{get_generic_question_template_answer(question, code, result, metadata).rstrip()}\n{AI_PROMPT}"
        return prompt
    elif LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ).rstrip(),
            }
        ]
        return system, prompt
    elif LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC,
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, code, result, metadata),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.StarCoderInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n{get_generic_question_template_answer(question, code, result,metadata)}"
        return prompt

    elif LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK}\n\n{get_deepseekcode_question_template_answer(question, code, result,metadata)}"
        return prompt
    elif LanguageModelStyle == LMStyle.CodeLLaMaInstruct:
        prompt = f"[INST] <<SYS>>\n{PromptConstants.SYSTEM_MESSAGE_GENERIC}\n<</SYS>>\n\n{get_cllama_question_template_answer(question, code, result,metadata)}\n[/INST]"
        return prompt
    # elif LanguageModelStyle == LMStyle.CodeQwenInstruct:
    #     # Handle Qwen models (including Qwen2.5-7B-Instruct)
    #     chat_messages = [
    #         {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
    #     ]
    #     chat_messages += [
    #         {
    #             "role": "user", 
    #             "content": get_generic_question_template_answer(
    #                 question, code, result, metadata
    #             ),
    #         },
    #     ]
        
    #     from transformers import AutoTokenizer
    #     tokenizer = AutoTokenizer.from_pretrained(
    #         "Qwen/Qwen2.5-7B-Instruct", padding_side="left", use_fast=False
    #     )
    #     return tokenizer.apply_chat_template(
    #         chat_messages,
    #         tokenize=False,
    #         add_generation_prompt=True,
    #         truncation=False,
    #         padding=False,
    #     )
    
    # Zero-shot CoT version of Self-repair
    elif LanguageModelStyle == LMStyle.CodeQwenInstruct:
        # Handle Qwen models (including Qwen2.5-7B-Instruct)
        system_message_qwen_cot_instruct = f"You are a helpful programming assistant and an expert Python programmer. You are helping a user write a program to solve a problem. The user has written some code, but it has some errors and is not passing the tests. Before you answer, think step by step about why the code fails and how to fix it. You will help the user by first giving a concise (at most 2-3 sentences) textual explanation of what is wrong with the code. After you have pointed out what is wrong with the code, you will then generate a fixed version of the program. You must put the entire fixed program within code delimiters only for once."
        system_message_qwen_scot_instruct = (
            "You are a helpful programming assistant and an expert Python programmer. "
            "You are helping a user debug a Python program that fails some tests.\n\n"
            "For each task, you MUST follow this structure:\n"
            "1. First, produce a structured debugging plan under the heading "
            "   '### Debugging Plan'.\n"
            "   - Briefly restate the problem.\n"
            "   - Analyze why the given code fails, using the failing input, wrong "
            "     output, expected output, and any error message.\n"
            "   - Describe the concrete changes needed to fix the bug.\n"
            "2. Then, under the heading '### Fixed Code', output the ENTIRE corrected "
            "   program in a single Python code block (one pair of ```python ... ``` only).\n"
            "IMPORTANT: Do not include any other fenced code blocks anywhere else in your response."
        )
        chat_messages = [
            {"role": "system", "content": system_message_qwen_scot_instruct},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_qwen_structured_cot_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]
        
        from transformers import AutoTokenizer
        # Prefer the Coder-Instruct tokenizer/template when using the coder model,
        # but fall back gracefully to the general Instruct template.
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-Coder-Instruct", padding_side="right", use_fast=True
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct", padding_side="right", use_fast=True
            )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    elif LanguageModelStyle == LMStyle.DracarysLlama:
        chat_messages = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(
                    question, code, result, metadata
                ),
            },
        ]

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "abacusai/Dracarys-Llama-3.1-70B-Instruct", padding_side="right", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )
    if LanguageModelStyle == LMStyle.Eurusx:
        prompt = "[INST] Write Python code to solve the task:\n"
        prompt += f"{get_wizard_question_template_answer(question, code, result,metadata)}"
        prompt += "[/INST]"
        return prompt
    else:
        raise NotImplementedError(
            f"LanguageModelStyle {LanguageModelStyle} not implemented"
        )


def extract_code(model_output: str, lmstyle: LMStyle):
    """
    Extract the repaired program from the model output.
    
    Robust version with comprehensive error handling to prevent crashes.
    """
    try:
        if not model_output or not isinstance(model_output, str):
            return ""
        
        lines = model_output.split("\n")
        if not lines:
            return ""

        # Collect all fenced code blocks with their ranges and language tag
        blocks = []  # (start_idx, end_idx, lang)
        i = 0
        while i < len(lines):
            try:
                line = lines[i].strip()
                if line.startswith("```"):
                    lang = line[3:].strip().lower()  # e.g., 'python'
                    # find closing fence
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith("```"):
                        j += 1
                    if j < len(lines):
                        blocks.append((i, j, lang))
                        i = j  # jump to closing fence
                i += 1
            except (IndexError, AttributeError) as e:
                # Skip malformed lines, continue processing
                i += 1
                continue

        if not blocks:
            return ""

        # 1) Prefer the block after a 'Fixed Code' heading
        try:
            fixed_heading_indices = [idx for idx, l in enumerate(lines) 
                                   if l and "fixed code" in l.lower()]
            if fixed_heading_indices:
                cutoff = fixed_heading_indices[-1]
                for start, end, lang in blocks:
                    if start > cutoff:
                        return "\n".join(lines[start + 1 : end])
        except (IndexError, AttributeError):
            pass  # Fall through to next strategy

        # 2) Prefer the last python block
        try:
            python_blocks = [b for b in blocks if len(b) >= 3 and b[2] in ("python", "py")]
            if python_blocks:
                start, end, _ = python_blocks[-1]
                return "\n".join(lines[start + 1 : end])
        except (IndexError, TypeError):
            pass  # Fall through to next strategy

        # 3) Fallback to the longest block
        try:
            if blocks:
                start, end, _ = max(blocks, key=lambda b: (b[1] - b[0]) if len(b) >= 2 else 0)
                return "\n".join(lines[start + 1 : end])
        except (ValueError, IndexError, TypeError):
            pass  # Fall through to final fallback

        # 4) Ultimate fallback: return empty string
        return ""
        
    except Exception as e:
        # Catch-all: log the error and return empty string
        print(f"Warning: extract_code failed with error {type(e).__name__}: {e}")
        return ""


def validate_environment():
    """Pre-flight checks to ensure environment is safe."""
    try:
        # Check imports
        import transformers
        print("✓ transformers available")
    except ImportError:
        print("⚠ transformers not available - some LM styles may fail")
    
    try:
        from lcb_runner.lm_styles import LMStyle
        print("✓ LMStyle import successful")
    except ImportError as e:
        print(f"✗ LMStyle import failed: {e}")
        return False
    
    # Check disk space for output files
    import shutil
    free_space = shutil.disk_usage("/tmp").free
    if free_space < 1024 * 1024:  # 1MB minimum
        print(f"⚠ Low disk space: {free_space} bytes available")
    else:
        print(f"✓ Disk space OK: {free_space // (1024*1024)} MB available")
    
    return True

def safe_test_extract_code():
    """Test extract_code function with various inputs to ensure robustness."""
    print("Testing extract_code function...")
    
    test_cases = [
        "",  # empty string
        "No code blocks here",  # no code
        "```python\nprint('hello')\n```",  # simple case
        "### Fixed Code\n```python\nprint('fixed')\n```",  # structured CoT
        "```\nno lang\n```\n```python\nprint('last')\n```",  # multiple blocks
    ]
    
    for i, test_input in enumerate(test_cases):
        try:
            result = extract_code(test_input, LMStyle.CodeQwenInstruct)
            print(f"  Test {i+1}: OK (returned {len(result)} chars)")
        except Exception as e:
            print(f"  Test {i+1}: ERROR - {e}")
            return False
    
    return True

def test():
    """Enhanced test function with comprehensive error handling."""
    print("=== LiveCodeBench Self-Repair Test ===")
    
    # Pre-flight checks
    if not validate_environment():
        print("Environment validation failed. Aborting.")
        return
    
    if not safe_test_extract_code():
        print("extract_code tests failed. Aborting.")
        return
    
    def write_str_or_json(prompt):
        try:
            if isinstance(prompt, str):
                fp.write(prompt)
            else:
                fp.write(json.dumps(prompt, indent=2))
        except Exception as e:
            print(f"Error writing output: {e}")
            fp.write(f"ERROR: {e}")
        return
    
    input_path = "output/GPT-3.5-Turbo-0125/Scenario.codegeneration_10_0.2_eval_all.json"
    if not os.path.exists(input_path):
        print(f"Input file not found: {os.path.abspath(input_path)}")
        print("Create a fake file or point the script to a valid evaluation JSON to proceed.")
        return

    # Test both baseline and structured CoT for comparison
    test_styles = [LMStyle.OpenAIChat, LMStyle.CodeQwenInstruct]
    
    for lm_style in test_styles:
        try:
            print(f"\nTesting {lm_style}...")
            
            with open(input_path) as f:
                try:
                    data = json.load(f)
                    if not data or not isinstance(data, list):
                        print(f"Invalid JSON structure in {input_path}")
                        continue
                    check_metadata = data[0]
                except Exception as e:
                    print(f"Failed to parse {input_path}: {e}")
                    continue

            # Safely extract metadata with defaults
            checked_base_question_cotent = check_metadata.get("question_content", "Default question")
            checked_base_codes = check_metadata.get("code_list", ["pass"])[0]
            checked_base_results = check_metadata.get("graded_list", [False])[0]
            checked_base_metadata = check_metadata.get("metadata", [{}])[0]

            print(f"  Question length: {len(str(checked_base_question_cotent))}")
            print(f"  Code length: {len(str(checked_base_codes))}")
            print(f"  Result: {checked_base_results}")

            # Generate prompt with error handling
            leetcode_prompt = format_prompt_self_repair(
                checked_base_question_cotent,
                lm_style,
                checked_base_codes,
                checked_base_results,
                checked_base_metadata,
            )

            # Write output safely
            out_path = f"/tmp/leetcode_{lm_style}.txt"
            try:
                with open(out_path, "w", encoding='utf-8') as fp:
                    write_str_or_json(leetcode_prompt)
            except Exception as e:
                print(f"Error writing {out_path}: {e}")
                continue

            if not leetcode_prompt:
                print(f"  Generated prompt is empty for {lm_style}")
                print(f"  This can happen when code already passed tests")
            else:
                prompt_len = len(str(leetcode_prompt))
                print(f"  ✓ Generated prompt ({prompt_len} chars) -> {out_path}")
                
                # Test extract_code on a sample output
                if isinstance(leetcode_prompt, str):
                    sample_output = f"### Fixed Code\n```python\nprint('test')\n```"
                    extracted = extract_code(sample_output, lm_style)
                    print(f"  ✓ extract_code test: {len(extracted)} chars extracted")
                    
        except Exception as e:
            print(f"  ✗ Error processing {lm_style}: {type(e).__name__}: {e}")
            continue
    
    print("\n=== Test completed safely ===")
    return


if __name__ == "__main__":
    test()
