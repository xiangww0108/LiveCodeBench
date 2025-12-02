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

    # --------------------
    # LOCALIZER (ChatML)
    # --------------------
    def build_localizer_messages(self, example):
        raw_code = example["code_list"][0]
        code_lines = raw_code.split("\n")
        numbered_code = "\n".join(
            f"{i+1} | {line}" for i, line in enumerate(code_lines)
        )

        # extract error
        error = "Unknown Error"
        if "metadata" in example and example["metadata"]:
            try:
                meta = json.loads(example["metadata"][0])
                error = meta.get("error_message", meta.get("error", "Unknown Error"))
            except:
                pass

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code bug localizer. "
                    "Return ONLY a JSON dict with two keys:\n"
                    "  - \"bug_span\": list of [start,end] line ranges\n"
                    "  - \"bug_summary\": short English explanation\n\n"
                    "Example:\n"
                    "{\n"
                    "  \"bug_span\": [[8,10]],\n"
                    "  \"bug_summary\": \"Loop range too small\"\n"
                    "}\n"
                    "Do NOT return anything except JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PROBLEM:\n{example['question_content']}\n\n"
                    f"ERROR:\n{error}\n\n"
                    f"CODE:\n{numbered_code}\n\n"
                    "Return ONLY the JSON dict."
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps({
                    "bug_span": example["label_bug_span"],
                    "bug_summary": example["label_bug_summary"],
                })
            }
        ]
        return messages

    # --------------------
    # PLANNER (plaintext)
    # --------------------
    def build_planner_prompt(self, example):
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
Produce 2-6 clear steps explaining how to fix the bug and then provide a corrected code snippet inside a Markdown code block.
Return plain text.

### Answer:
"""

        target = example["label_plan"]
        return prompt, target

    # --------------------
    # MAIN
    # --------------------
    def __getitem__(self, idx):
        example = self.data[idx]
        task = example["task"]

        # ============================
        # LOCALIZER (CHATML)
        # ============================
        if task == "localizer":
            messages = self.build_localizer_messages(example)

            # build input (system + user + assistant)
            enc = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )

            input_ids = enc[0]

            # === create labels ===
            # mask system + user parts
            # let tokenizer tell us where 'assistant' starts
            assistant_enc = self.tokenizer.apply_chat_template(
                messages[:-1] + [{"role": "assistant", "content": ""}],
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )[0]

            assistant_start = assistant_enc.shape[0]

            labels = input_ids.clone()
            labels[:assistant_start] = -100  # mask system+user

            return {
                "input_ids": input_ids,
                "labels": labels
            }

        # ============================
        # PLANNER (PLAINTEXT)
        # ============================
        elif task == "planner":
            prompt, target = self.build_planner_prompt(example)

            full_text = prompt + target

            enc = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            input_ids = enc["input_ids"][0]

            labels = input_ids.clone()
            prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"][0]

            labels[: len(prompt_ids)] = -100

            return {
                "input_ids": input_ids,
                "labels": labels,
            }

        else:
            raise ValueError("Unknown task")
