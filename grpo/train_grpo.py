"""
GRPO (Group Relative Policy Optimization) training loop.

Uses TRL's GRPOTrainer which handles:
  - Rollout sampling (G responses per problem)
  - KL penalty against frozen reference model
  - Policy gradient update with PPO-style clipping

Reference model is loaded in 8-bit 

"""

import os
import sys
import json
import yaml
import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import GRPOTrainer, GRPOConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from grpo.reward import batch_rewards
from utils.vram_utils import print_vram, VRAMGuard, free_vram

SYSTEM_PROMPT = (
    "You are an expert math olympiad solver. "
    "Think step by step. "
    "At the end, state your final answer as a single integer on a new line prefixed with 'Answer:'."
)


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "configs/grpo_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_grpo_dataset(path: str, tokenizer) -> Dataset:
    with open(path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    rows = []
    for rec in records:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": rec["problem"].strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        rows.append({
            "prompt":       prompt,
            "ground_truth": str(rec["answer"]).strip(),
        })

    return Dataset.from_list(rows)


# ── Reward wrapper ────────────────────────────────────────────────────────────

def make_reward_fn():
    """
    Returns a reward function compatible with TRL's GRPOTrainer.
    Signature expected by TRL: fn(prompts, completions, **kwargs) -> list[float]
    """
    def reward_fn(prompts, completions, ground_truth=None, **kwargs):
        rewards = []
        for completion, gt in zip(completions, ground_truth):
            r = batch_rewards([completion], gt)[0]
            rewards.append(r)
        return rewards

    return reward_fn


# ── Model loading ─────────────────────────────────────────────────────────────

def load_policy_model(cfg: dict):
    path = cfg["model"]["policy_path"]
    dtype = torch.bfloat16 if cfg["model"]["dtype"] == "bfloat16" else torch.float16

    with VRAMGuard("loading policy model"):
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
    return model


def load_ref_model(cfg: dict):
    """Load reference model in 8-bit to save VRAM."""
    path = cfg["model"]["ref_path"]
    use_8bit = cfg["model"].get("ref_load_in_8bit", True)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True) if use_8bit else None

    with VRAMGuard("loading reference model (8-bit)"):
        ref_model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

    # Freeze reference model
    for param in ref_model.parameters():
        param.requires_grad = False

    return ref_model


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    grpo_cfg = cfg["grpo"]
    train_cfg = cfg["training"]

    wandb.init(project="aimo-grpo", config=cfg)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["policy_path"], trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for generation

    # Load models
    policy_model = load_policy_model(cfg)
    ref_model    = load_ref_model(cfg)
    print_vram("after loading both models")

    # Dataset
    dataset = load_grpo_dataset(cfg["data"]["train_path"], tokenizer)
    print(f"GRPO dataset: {len(dataset):,} problems")

    # GRPO config
    grpo_training_args = GRPOConfig(
        output_dir=train_cfg["output_dir"],
        max_steps=train_cfg["max_steps"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        report_to=train_cfg["report_to"],
        # GRPO-specific
        num_generations=grpo_cfg["group_size"],           # G
        max_new_tokens=grpo_cfg["max_new_tokens"],
        temperature=grpo_cfg["temperature"],
        top_p=grpo_cfg["top_p"],
        kl_coeff=grpo_cfg["kl_coeff"],
        cliprange=grpo_cfg["clip_range"],
    )

    trainer = GRPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        args=grpo_training_args,
        train_dataset=dataset,
        reward_funcs=make_reward_fn(),
    )

    print("\nStarting GRPO training...")
    print(f"  Group size G:  {grpo_cfg['group_size']}")
    print(f"  KL coeff β:    {grpo_cfg['kl_coeff']}")
    print(f"  Max steps:     {train_cfg['max_steps']}")
    print_vram("before training")

    trainer.train()

    print("\nSaving final GRPO checkpoint...")
    trainer.save_model(train_cfg["output_dir"] + "/final")
    tokenizer.save_pretrained(train_cfg["output_dir"] + "/final")
    print("GRPO training complete.")


if __name__ == "__main__":
    main()