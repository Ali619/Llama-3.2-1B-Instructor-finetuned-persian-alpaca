import os
from multiprocessing import Pool

import datasets
import pandas as pd
import torch
import wandb
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


def process_batch(batch):
    data_df = pd.DataFrame(batch)
    data_df["text"] = data_df[["instruction", "input", "output"]].apply(
        lambda x: "<|im_start|>user\n"
        + x["instruction"]
        + ":\n"
        + x["input"]
        + "<|im_end|>\n<|im_start|>assistant\n"
        + x["output"]
        + "<|im_end|>\n",
        axis=1,
    )
    return Dataset.from_pandas(data_df[["text"]])


def prepare_train_data(data, batch_size=50000) -> Dataset:
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    with Pool(processes=10) as pool:
        processed_datasets = pool.map(process_batch, batches)

    return datasets.concatenate_datasets(processed_datasets)


data = load_dataset("mshojaei77/merged_persian_alpaca", split="train")
processed_data = prepare_train_data(data)
del data

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
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
WB_ENTITY = os.getenv("WB_ENTITY")
wandb.init(
    entity=WB_ENTITY,
    project="Llama-3-2-1B-Instructor-finetuned-persian-alpaca",
    resume="auto",
)

OUTPUT_DIR = "./output/Llama-3-2-1B-Instructor-finetuned-persian-alpaca"
HF_MODEL_NAME = "ali619/Llama-3.2-1B-Instruct-finetune-persian-alpaca"
STEP = 100

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=0.5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    optim="adamw_bnb_8bit",
    learning_rate=2e-3,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    bf16=True,
    logging_steps=STEP,
    save_steps=100,
    save_strategy="steps",
    save_total_limit=2,
    hub_strategy="checkpoint",
    report_to=["wandb", "tensorboard"],
    dataloader_pin_memory=True,
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
    max_seq_length=2048,
)

trainer.train(resume_from_checkpoint=True)
