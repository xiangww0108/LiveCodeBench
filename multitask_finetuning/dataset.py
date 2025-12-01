import json
import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_length=2048):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def make_localizer_messages(self, example):
        raw_code = example["code_list"][0]
        code_lines = raw_code.split("\n")
        numbered_code = "\n".join(f"{i+1} | {line}" for i, line in enumerate(code_lines))

        meta_str = example["metadata"][0]
        try:
            meta_json = json.loads(meta_str)
            error = meta_json.get("error_message", "Unknown Error")
        except:
            error = "Unknown Error"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code debugger. Locate the bug. "
                    "Output ONLY a JSON list of line ranges like [[start, end]]."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PROBLEM:\n{example['question_content']}\n\n"
                    f"ERROR:\n{error}\n\n"
                    f"CODE:\n{numbered_code}"
                )
            },
            {
                "role": "assistant",
                "content": json.dumps(example["label_bug_span"])
            }
        ]
        return messages

    def make_planner_prompt(self, example):
        prompt = f"""### Task: planner
### Problem:
{example['question_content']}

### Code:
{example['code_list'][0]}

### Bug Span:
{example['bug_span']}

### Bug Summary:
{example['bug_summary']}

### What to do:
Produce 2-6 steps explaining how to fix the bug and provide a corrected snippet.
Return plain text.

### Answer:
"""
        target = example["label_plan"]
        return prompt, target

    def __getitem__(self, idx):
        example = self.data[idx]
        task = example["task"]

        if task == "localizer":
            messages = self.make_localizer_messages(example)
            encoded = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            # 1D
            input_ids = encoded[0]
            return {
                "input_ids": input_ids,
                "labels": input_ids.clone()
            }

        elif task == "planner":
            prompt, target = self.make_planner_prompt(example)
            encoding = self.tokenizer(
                prompt + target,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"][0]  # 1D
        
            labels = input_ids.clone()
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
            labels[: len(prompt_ids)] = -100  # mask prompt

            return {
                "input_ids": input_ids,
                "labels": labels
            }

        else:
            raise ValueError("Unknown task")
