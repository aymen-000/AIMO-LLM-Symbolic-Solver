"""
SFT fine-tuning for Qwen2.5 using LoRA.
Uses HuggingFace TRL's SFTTrainer (updated for v0.11.0+).

"""

import os
import sys
import json
import yaml
import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Try to import VRAM utilities, but provide stubs if not available
try:
    from utils.vram_utils import print_vram, VRAMGuard
except ImportError:
    from contextlib import contextmanager
    
    def print_vram(msg):
        print(f"[VRAM] {msg}")
    
    @contextmanager
    def VRAMGuard(msg):
        print(f"[VRAMGuard] {msg}")
        yield
        print(f"[VRAMGuard] Done: {msg}")


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/sft_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dataset(path: str, tokenizer, max_seq_length: int) -> Dataset:
    """Load and prepare dataset from JSONL file with chat messages."""
    records = load_jsonl(path)
    texts = []
    for rec in records:
        # Apply chat template to format messages
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})
    return Dataset.from_list(texts)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    """Load model and tokenizer with LoRA configuration."""
    model_name = cfg["model"]["name"]
    dtype = torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float16

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"Loading model from {model_name}...")
    with VRAMGuard("loading base model"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  
        )

    # Apply LoRA if enabled
    if cfg["lora"]["enabled"]:
        print("Applying LoRA configuration...")
        lora_cfg = cfg["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            target_modules=lora_cfg["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    # Required for PEFT + gradient checkpointing
    if cfg["training"]["gradient_checkpointing"]:
        model.enable_input_require_grads()

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    train_cfg = cfg["training"]

    wandb.init(project="aimo-sft", config=cfg)

    # Load model and tokenizer
    print("\n" + "="*60)
    print("LOADING MODEL AND TOKENIZER")
    print("="*60)
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Build datasets
    print("\n" + "="*60)
    print("BUILDING DATASETS")
    print("="*60)
    train_dataset = build_dataset(
        cfg["data"]["train_path"], tokenizer, cfg["model"]["max_seq_length"]
    )
    eval_dataset = build_dataset(
        cfg["data"]["eval_path"], tokenizer, cfg["model"]["max_seq_length"]
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    # Calculate warmup steps (recommended over warmup_ratio in newer versions)
    total_train_steps = (
        (len(train_dataset) // train_cfg["per_device_train_batch_size"]) 
        * train_cfg["num_train_epochs"]
    ) // train_cfg["gradient_accumulation_steps"]
    warmup_steps = int(total_train_steps * train_cfg.get("warmup_ratio", 0.1))

    print(f"Total training steps: {total_train_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # TrainingArguments - updated for latest transformers API
    print("\n" + "="*60)
    print("CREATING TRAINING ARGUMENTS")
    print("="*60)
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,  # Use warmup_steps instead of warmup_ratio
        weight_decay=train_cfg["weight_decay"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        adam_epsilon=train_cfg["adam_epsilon"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy=train_cfg["logging_strategy"],
        logging_steps=train_cfg.get("logging_steps", 10),
        save_total_limit=train_cfg["save_total_limit"],
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        report_to=train_cfg.get("report_to", ["wandb"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Initialize SFTTrainer
    # IMPORTANT: Updated API for trl >= 0.11.0
    # - NO tokenizer parameter
    # - NO data_collator parameter  
    # - NO max_seq_length parameter
    print("\n" + "="*60)
    print("INITIALIZING TRAINER")
    print("="*60)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    # Set tokenizer after initialization
    trainer.tokenizer = tokenizer

    # Start training
    print_vram("before training")
    print("\n" + "="*60)
    print("STARTING SFT TRAINING")
    print("="*60 + "\n")
    trainer.train()

    # Save final model
    print("\n" + "="*60)
    print("SAVING FINAL CHECKPOINT")
    print("="*60)
    final_output_dir = os.path.join(train_cfg["output_dir"], "final")
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"Model saved to {final_output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()