"""
Merges LoRA adapter weights into the base model to produce
a single full-weight checkpoint ready for GRPO.

Usage:
    python merge.py --config config.yaml
"""

import os
import sys
import argparse
import yaml
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.vram_utils import VRAMGuard, print_vram


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--adapter_path", type=str, default=None, help="Override adapter path")
    parser.add_argument("--output_path", type=str, default=None, help="Override output path")
    args = parser.parse_args()

    config = load_config(args.config)

    # =========================
    # Extract from config
    # =========================
    BASE_MODEL = config["model"]["name"]
    DTYPE_STR  = config["model"]["dtype"]

    TRAIN_OUT  = config["training"]["output_dir"]

    # Defaults (can override via CLI)
    ADAPTER_PATH = args.adapter_path or os.path.join(TRAIN_OUT, "final")
    OUTPUT_PATH  = args.output_path or TRAIN_OUT + "_merged"

    # dtype mapping
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    TORCH_DTYPE = dtype_map.get(DTYPE_STR, torch.bfloat16)

    print(f"Base model:    {BASE_MODEL}")
    print(f"LoRA adapter:  {ADAPTER_PATH}")
    print(f"Output:        {OUTPUT_PATH}")
    print(f"Dtype:         {TORCH_DTYPE}\n")

    # =========================
    # Load base model
    # =========================
    with VRAMGuard("loading base model"):
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=TORCH_DTYPE,
            device_map="auto",
            trust_remote_code=True,
        )

    # =========================
    # Merge LoRA
    # =========================
    with VRAMGuard("merging LoRA"):
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = model.merge_and_unload()
        print("LoRA merged successfully.")

    print_vram("after merge")

    # =========================
    # Save merged model
    # =========================
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print(f"\nSaving merged model to {OUTPUT_PATH}...")
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("\n Merge complete.")
    print(f"Merged checkpoint: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()