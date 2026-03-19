"""
SFT fine-tuning for Qwen2.5-Math-7B-Instruct using LoRA.
Uses HuggingFace TRL's SFTTrainer.

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
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.vram_utils import print_vram, VRAMGuard


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/sft_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_dataset(path: str, tokenizer, max_seq_length: int) -> Dataset:
    records = load_jsonl(path)
    texts = []
    for rec in records:
        text = tokenizer.apply_chat_template(
            rec["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append({"text": text})
    return Dataset.from_list(texts)


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: dict):
    model_name = cfg["model"]["name"]
    dtype = torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with VRAMGuard("loading base model"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",  
        )

    if cfg["lora"]["enabled"]:
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
        
    # required for older PEFT + gradient checkpointing
    if cfg["training"]["gradient_checkpointing"]:
        model.enable_input_require_grads()

    return model, tokenizer


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    train_cfg = cfg["training"]

    wandb.init(project="aimo-sft", config=cfg)

    model, tokenizer = load_model_and_tokenizer(cfg)

    print("\nBuilding datasets...")
    train_dataset = build_dataset(
        cfg["data"]["train_path"], tokenizer, cfg["model"]["max_seq_length"]
    )
    eval_dataset = build_dataset(
        cfg["data"]["eval_path"], tokenizer, cfg["model"]["max_seq_length"]
    )

    # Only compute loss on assistant tokens
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        adam_beta1=train_cfg["adam_beta1"],
        adam_beta2=train_cfg["adam_beta2"],
        adam_epsilon=train_cfg["adam_epsilon"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy=train_cfg["save_strategy"],
        eval_steps=train_cfg["eval_steps"],
        logging_strategy=train_cfg["logging_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
        report_to=train_cfg["report_to"],
        evaluation_strategy=train_cfg["evaluation_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        max_seq_length=cfg["model"]["max_seq_length"],
        dataset_text_field="text",
    )

    print_vram("before training")
    print("\nStarting SFT training...")
    trainer.train()

    print("\nSaving final checkpoint...")
    trainer.save_model(train_cfg["output_dir"] + "/final")
    tokenizer.save_pretrained(train_cfg["output_dir"] + "/final")
    print("Done.")


if __name__ == "__main__":
    main()