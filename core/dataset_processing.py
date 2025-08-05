import os
import json
from datasets import load_dataset

dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1", split="train")

formatted = []
for example in dataset:
    formatted.append({
        "instruction": "Extract the structured receipt information from the OCR text.",
        "input": example["parsed_data"],
        "output": json.dumps(example["raw_data"], ensure_ascii=False)
    })

os.makedirs("data", exist_ok=True)
with open("data/train.jsonl", "w", encoding="utf-8") as f:
    for item in formatted:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
