"""
Merges LoRA adapter weights into the base model to produce
a single full-weight checkpoint ready for GRPO.
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.vram_utils import VRAMGuard, print_vram

ADAPTER_PATH = "./outputs/sft/final"
BASE_MODEL   = "Qwen/Qwen2.5-Math-7B-Instruct"
OUTPUT_PATH  = "./outputs/sft_merged"


def main():
    print(f"Base model:    {BASE_MODEL}")
    print(f"LoRA adapter:  {ADAPTER_PATH}")
    print(f"Output:        {OUTPUT_PATH}\n")

    # Load base model
    with VRAMGuard("loading base model"):
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    # Load and merge LoRA
    with VRAMGuard("merging LoRA"):
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model = model.merge_and_unload()
        print("LoRA merged successfully.")

    print_vram("after merge")

    # Save merged model
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    print(f"\nSaving merged model to {OUTPUT_PATH}...")
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, trust_remote_code=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    print("Merge complete.")
    print(f"Merged checkpoint: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()