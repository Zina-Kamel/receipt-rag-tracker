import os
import json
import mlflow
import ast
import logging
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("lora-mistral-finetuning")

def clean_json_str(s):
    """Normalize JSON-like strings by stripping whitespace and ensuring double quotes instead of single."""
    if not isinstance(s, str):
        return ""
    return s.strip().replace("'", '"')
    
def robust_parse_json(s):
    """
    Try parsing a JSON string using json.loads first, then ast.literal_eval as fallback.
    Returns an empty dict on failure.
    """
    if not isinstance(s, str):
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(s)
        except Exception as e:
            print(f"Failed to parse structured JSON: {e}")
            return {}

def parse_structured_json(raw_input):
    """
    Prepares dataset for training by extracting nested 'json' field if present and parse it.
    """
    try:
        outer = robust_parse_json(raw_input)
        if isinstance(outer, dict) and "json" in outer:
            nested = robust_parse_json(outer["json"])
            return nested if nested else outer["json"]
        return outer
    except Exception as e:
        print(f"Failed to parse structured JSON: {e}")
        return {}

def extract_ocr_words(raw_output):
    """
    Extract OCR words from JSON output.
    """
    if not isinstance(raw_output, str):
        return ""
    
    json_start = raw_output.find('{"')
    json_end = raw_output.rfind('"}') + 2
    
    if json_start != -1 and json_end > json_start:
        json_str = raw_output[json_start:json_end]
    else:
        json_str = raw_output
    
    try:
        outer = json.loads(json_str)
    except json.JSONDecodeError:
        return raw_output

    if isinstance(outer, dict):
        ocr_words = outer.get("ocr_words", [])
    else:
        outer = json.loads(outer)
        outer = outer["ocr_words"]
    
    return str(outer)

def build_prompt(row):
    """
    Construct training prompt from OCR text and structured JSON.
    Example format:
        ### OCR Words:
        word1 word2 ...

        ### Structured JSON:
        {...}
    """
    structured_json_str = json.dumps(row["structured_json"], indent=2, ensure_ascii=False)
    schema = """{
    "header": {
        "invoice_no": "string",
        "invoice_date": "MM/DD/YYYY",
        "seller": "string (name + address)",
        "client": "string (name + address)",
        "seller_tax_id": "string",
        "client_tax_id": "string",
        "iban": "string"
    },
    "items": [
        {
        "item_desc": "string",
        "item_qty": "number",
        "item_net_price": "number",
        "item_net_worth": "number",
        "item_vat": "percentage",
        "item_gross_worth": "number"
        }
    ],
    "summary": {
        "total_net_worth": "currency amount",
        "total_vat": "currency amount",
        "total_gross_worth": "currency amount"
    }
    }"""

    return (
        f"### Task: Convert OCR extracted words from invoices into structured JSON format\n\n"
        f"### OCR Words:\n{extract_ocr_words(row['ocr_text'])}\n\n"
        f"### Expected JSON Schema:\n{schema}\n\n"
        f"### Structured JSON:\n{structured_json_str}"
    )

def analyze_prompt_lengths(df):
    """Analyze the distribution of prompt lengths to optimize max_length"""
    lengths = []
    for prompt in df["prompt"]:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        lengths.append(len(tokens))
    
    lengths_series = pd.Series(lengths)
    print(f"Prompt length statistics:")
    print(f"Mean: {lengths_series.mean():.1f}")
    print(f"Median: {lengths_series.median():.1f}")
    print(f"95th percentile: {lengths_series.quantile(0.95):.1f}")
    print(f"Max: {lengths_series.max()}")
    
    recommended_length = int(lengths_series.quantile(0.95))
    print(f"Recommended max_length: {recommended_length}")
    return recommended_length

logger.info("Loading dataset...")
df = pd.read_json("data/train.jsonl", lines=True)

logger.info("Extracting OCR text and structured JSON...")
df["ocr_text"] = df["output"].apply(extract_ocr_words)
df["structured_json"] = df["input"].apply(parse_structured_json)

logger.info("Building prompts for fine-tuning...")
df["prompt"] = df.apply(build_prompt, axis=1)

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

logger.info("Analyzing prompt lengths...")
optimal_max_length = analyze_prompt_lengths(df)

USE_DYNAMIC_PADDING = True  

def tokenize_fn(example):
    if USE_DYNAMIC_PADDING:
        enc = tokenizer(
            example["prompt"],
            truncation=True,
            padding=False, 
            max_length=optimal_max_length,
        )
    else:
        enc = tokenizer(
            example["prompt"],
            truncation=True,
            padding="max_length",
            max_length=min(optimal_max_length, 1024),  
        )
    
    enc["labels"] = enc["input_ids"].copy()
    return enc

dataset = Dataset.from_pandas(df[["prompt"]])
dataset = dataset.train_test_split(test_size=0.1)

logger.info("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_fn, batched=False)

keep_cols = {"input_ids", "attention_mask", "labels"}
for split in ["train", "test"]:
    tokenized_dataset[split] = tokenized_dataset[split].remove_columns(
        set(tokenized_dataset[split].column_names) - keep_cols
    )

example = tokenized_dataset["train"][0]
print(f"Example tokenization:")
print(f"Input IDs length: {len(example['input_ids'])}")
print(f"Attention mask sum: {sum(example['attention_mask']) if 'attention_mask' in example else 'N/A'}")
print(f"Padding ratio: {(len(example['input_ids']) - sum(example['attention_mask'])) / len(example['input_ids']) * 100:.1f}%" 
      if 'attention_mask' in example else 'N/A')
print(f"First few tokens: {example['input_ids'][:20]}")

if USE_DYNAMIC_PADDING:
    from transformers import DataCollatorForLanguageModeling
    print("Using dynamic padding")

print(tokenized_dataset["train"][0])

logger.info("Loading base model with 8-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    llm_int8_enable_fp32_cpu_offload=True,  
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",  
    quantization_config=bnb_config,
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.train()

mlflow.log_params({
    "model_name": model_name,
    "r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "lora_dropout": lora_config.lora_dropout,
    "learning_rate": 2e-4,
    "batch_size": 1,
    "epochs": 1,
    "max_length": 1024,
    "quantization": "8-bit",
})

training_args = TrainingArguments(
    output_dir="./lora-mistral-json-output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  
    num_train_epochs=1,
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="steps",   
    eval_steps=50,                  
    logging_dir="./logs",
    fp16=True,                      
    dataloader_pin_memory=False,    
    remove_unused_columns=False,    
    report_to="mlflow",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
    args=training_args,
)

logger.info("Starting training...")
trainer.train()

logger.info("Evaluating model...")
eval_metrics = trainer.evaluate()
mlflow.log_metrics(eval_metrics)

logger.info("Saving fine-tuned model and tokenizer...")
trainer.model.save_pretrained("./lora-mistral-json-output")
tokenizer.save_pretrained("./lora-mistral-json-output")
mlflow.log_artifacts("./lora-mistral-json-output", artifact_path="model")
