import json
from torch.utils.data import Dataset
from prompt import build_prompt

class MultiTaskDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=2048):
        self.data = json.loads(open(path).read())
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt, target = build_prompt(sample)
        full_text = prompt + "\n" + target

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
        )

        # label masking
        labels = enc["input_ids"].copy()
        # Let model learn full supervised loss
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
