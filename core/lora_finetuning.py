import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model_name = "mistralai/Mistral-7B-v0.1"
output_dir = "./lora-mistral-json-output"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

if torch.cuda.is_available():
    model = model.to("cuda")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v1", split="train")

def format_example(example):
    prompt = f"""### OCR Text:
{example['parsed_data']}

### Extracted JSON:
{example['raw_data']}
"""
    return {"text": prompt}

dataset = dataset.map(format_example)
dataset = dataset.train_test_split(test_size=0.1)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=1024)

tokenized_dataset = dataset.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(
    set(tokenized_dataset["train"].column_names) - {"input_ids", "attention_mask"}
)

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    logging_dir="./logs",
    bf16=False,
    fp16=False,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_args,
    dataset_text_field="text",
)

trainer.train()

trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
