import json

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
    outputlines = model_output.split("\n")
    if lmstyle == LMStyle.CodeLLaMa:
        indexlines = [i for i, line in enumerate(outputlines) if "PYTHON]" in line]
    else:
        indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def test():
    def write_str_or_json(prompt):
        if isinstance(prompt, str):
            fp.write(prompt)
        else:
            fp.write(json.dumps(prompt))
        return

    for lm_style in [LMStyle.OpenAIChat]:
        with open(
            "output/GPT-3.5-Turbo-0125/Scenario.codegeneration_10_0.2_eval_all.json"
        ) as f:
            check_metadata = json.load(f)[0]
        checked_base_question_cotent = check_metadata["question_content"]
        checked_base_codes = check_metadata["code_list"][0]
        checked_base_results = check_metadata["graded_list"][0]
        checked_base_metadata = check_metadata["metadata"][0]
        leetcode_prompt = format_prompt_self_repair(
            checked_base_question_cotent,
            lm_style,
            checked_base_codes,
            checked_base_results,
            checked_base_metadata,
        )

        with open(f"/tmp/leetcode_{lm_style}.txt", "w") as fp:
            write_str_or_json(leetcode_prompt)
    return


if __name__ == "__main__":
    test()
