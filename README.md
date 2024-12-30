### Fine-Tune Llama 3.2-1B with only 3 GiB of gpu memory and batch_size=8

[*unsloth*](https://github.com/unslothai/unsloth) will reduce a large amount of VRAM usage and also speedup training even on low-rank consumer GPU devices. I'm using a **Nvidia RTX-3060** with *12 GB* VRAM to fine-tune this model and just used **2.5 GB of VRAM** with `batch siez=8` per device. You can see all the hyperparameters below:

```python 
from transformers import TrainingArguments

STEP = 100
BATCH_SIZE = 8
MAX_LENGTH = 2048

per_device_train_batch_size=BATCH_SIZE
gradient_accumulation_steps=int(16 / BATCH_SIZE)
optim="adamw_bnb_8bit"
learning_rate=2e-3
lr_scheduler_type="cosine"
weight_decay=0.01
warmup_steps=STEP
fp16=not is_bfloat16_supported()
bf16=is_bfloat16_supported() # True
```

## How to run

1. Install `requirements.txt`:
```
pip install -r requirements.txt
```
> **Note:** You have to install *cuda* and other dependencies to run this script on GPU, otherwise it would be too slow to finish.

2. Run:
```
python fine-tune.py
```