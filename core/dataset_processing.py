import os
import json
from datasets import load_dataset

from config import settings

DATASET_NAME = settings.FINETUNE_INVOICES_DATASET
DATA_DIR = settings.DATA_DIR

dataset = load_dataset(DATASET_NAME, split="train")

formatted = []
for example in dataset:
    formatted.append({
        "instruction": "Extract the structured receipt information from the OCR text.",
        "input": example["parsed_data"],
        "output": json.dumps(example["raw_data"], ensure_ascii=False)
    })

os.makedirs(DATA_DIR, exist_ok=True)
file_path = os.path.join(DATA_DIR, "train.jsonl")

with open(file_path, "w", encoding="utf-8") as f:
    for item in formatted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")