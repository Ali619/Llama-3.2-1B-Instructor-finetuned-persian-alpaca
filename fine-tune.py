import os

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)
# model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.pretraining_tp = 1


def prepreprocess_batch(batch):
    text = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        + instruct
        + ":\n"
        + _input
        + "<|eot_id|>\n"
        + "<|start_header_id|>assistant<|end_header_id|>"
        + output
        + "<|eot_id|>"
        for instruct, _input, output in zip(
            batch["instruction"], batch["input"], batch["output"]
        )
    ]
    return {"text": text}


data = load_dataset("mshojaei77/merged_persian_alpaca", split="train")
processed_data = data.map(prepreprocess_batch, batched=True, num_proc=os.cpu_count())
del data

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    use_rslora=True,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

OUTPUT_DIR = "./output/Llama-3-2-1B-Instructor-finetuned-persian-alpaca"
HF_MODEL_NAME = "ali619/Llama-3.2-1B-Instruct-finetune-persian-alpaca"
STEP = 100
MAX_LENGTH = 2048

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=0.5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=int(STEP / 10),
    optim="adamw_bnb_8bit",
    learning_rate=2e-3,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    logging_steps=STEP,
    save_steps=STEP,
    save_strategy="steps",
    save_total_limit=2,
    hub_strategy="checkpoint",
    report_to=["tensorboard"],
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    hub_model_id=HF_MODEL_NAME,
    push_to_hub=True,
    seed=42,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=processed_data,
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=MAX_LENGTH,
)

trainer.train()
