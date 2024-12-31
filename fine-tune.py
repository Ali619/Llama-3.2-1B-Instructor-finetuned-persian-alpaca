import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"

MAX_LENGTH = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=MAX_LENGTH,
    load_in_4bit=True,
    dtype=None,
)
tokenizer.pad_token = tokenizer.eos_token
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "up_proj",
        "down_proj",
        "o_proj",
        "gate_proj",
    ],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
    random_state=42,
    loftq_config=None,
)
print(model.print_trainable_parameters())


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
processed_data = data.map(prepreprocess_batch, batched=True)
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
BATCH_SIZE = 32
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=2,
    optim="adamw_bnb_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    warmup_steps=STEP,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
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
    packing=True,  # can make training 5x faster for short sequences
)

trainer.train()
