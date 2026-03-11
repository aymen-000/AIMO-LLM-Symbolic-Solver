"""
Downloads NuminaMath-CoT, MATH, and AIME datasets from HuggingFace.
Saves raw splits to ./data/raw/
"""

import os
from datasets import load_dataset
from tqdm import tqdm
import json

RAW_DIR = "./data/raw"
os.makedirs(RAW_DIR, exist_ok=True)


def save_jsonl(records: list[dict], path: str):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records):,} records → {path}")


def download_numina():
    """NuminaMath-CoT — ~860K competition math problems with CoT traces."""
    print("Downloading NuminaMath-CoT...")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    records = []
    for row in tqdm(ds):
        records.append({
            "source": "numina",
            "problem": row["problem"],
            "solution": row["solution"],   # full CoT trace
            "answer": row.get("answer", ""),
        })
    save_jsonl(records, f"{RAW_DIR}/numina.jsonl")

def download_numina_tir():
    ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    records = []
    for row in tqdm(ds):
        records.append({
            "source": "numina_tir",
            "problem": row["problem"],
            "solution": row["solution"],   # already has code blocks
            "answer": row.get("answer", ""),
        })
    save_jsonl(records, f"{RAW_DIR}/numina_tir.jsonl")
    
def download_math():
    """Hendrycks MATH dataset — 12.5K problems, levels 1–5."""
    print("Downloading MATH dataset...")
    ds = load_dataset("lighteval/MATH", "all", split="train")
    records = []
    for row in tqdm(ds):
        records.append({    
            "source": "math",
            "problem": row["problem"],
            "solution": row["solution"],
            "answer": row["solution"].split("\\boxed{")[-1].rstrip("}") if "\\boxed{" in row["solution"] else "",
            "level": row.get("level", ""),
            "type": row.get("type", ""),
        })
    save_jsonl(records, f"{RAW_DIR}/math.jsonl")


def download_aime():
    """AIME problems 1983–2024."""
    print("Downloading AIME dataset...")
    ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
    records = []
    for row in tqdm(ds):
        records.append({
            "source": "aime",
            "problem": row["problem"],
            "solution": row.get("solution", ""),
            "answer": str(row.get("answer", "")),
        })
    save_jsonl(records, f"{RAW_DIR}/aime.jsonl")


if __name__ == "__main__":
    download_numina()
    download_math()
    download_aime()
    download_numina_tir()
    print("\nAll downloads complete.")