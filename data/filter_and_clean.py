"""
Filters and cleans raw datasets for SFT training.

Rules:
  - Deduplicate by problem text hash
  - Format into chat template for Qwen2.5-Math-7B-Instruct
  - Extract answer from \\boxed{} if no explicit answer field
  - 95/5 train/eval split
"""

import json
import hashlib
import os
import re
from tqdm import tqdm
import argparse
import yaml
import random
RAW_DIR = "./data/raw"
OUT_DIR = "./data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert math olympiad solver. "
    "Think step by step. "
    "Always express your solution as executable Python code. "
    "The code should compute the final answer directly. "
    "Do not include explanations outside the code. "
    "At the end, print the final answer in the format: print('Answer:', result)"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str:
    """Extract the last \\boxed{...} value from a solution string."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return matches[-1].strip() if matches else ""


def get_field(rec: dict, *candidates) -> str:
    """Try multiple field name candidates, return first non-empty match."""
    for key in candidates:
        if key in rec and rec[key]:
            return str(rec[key]).strip()
    return ""


# ── Formatting ────────────────────────────────────────────────────────────────

def format_chat(problem: str, solution: str, answer: str) -> dict:
    """Format a sample into Qwen2.5 chat template."""
    assistant_content = solution.strip()

    # Only append Answer: line if not already present
    if answer and f"\\boxed{{{answer}}}" not in assistant_content:
        if "Answer:" not in assistant_content:
            assistant_content += f"\n\nAnswer: {answer}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": problem.strip()},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records):,} → {path}")


# ── Dedup ─────────────────────────────────────────────────────────────────────

def deduplicate(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for rec in records:
        h = hashlib.md5(rec["problem"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(rec)
    return out


# ── Processing ────────────────────────────────────────────────────────────────

def process_records(records: list[dict], source: str) -> list[dict]:
    """Normalize field names — no filtering, just format everything."""
    out = []
    for rec in tqdm(records, desc=f"Processing {source}"):
        problem  = get_field(rec, "problem", "question", "input", "prompt")
        solution = get_field(rec, "solution", "reasoning", "chain_of_thought",
                             "rationale", "response", "output", "text")

        # Try explicit answer field first, fall back to extracting from solution
        answer = get_field(rec, "answer", "final_answer", "label", "target")
        if not answer:
            answer = extract_boxed(solution)

        # Only skip if there's literally no question
        if not problem:
            continue

        rec["problem"]  = problem
        rec["solution"] = solution
        rec["answer"]   = answer
        out.append(rec)

    print(f"  {source}: {len(out):,} records")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    all_records = []

    for fname in ["numina.jsonl", "math.jsonl", "aime.jsonl"]:
        path = f"{RAW_DIR}/{fname}"
        if not os.path.exists(path):
            print(f"Skipping {fname} (not found)")
            continue
        raw = load_jsonl(path)
        print(f"Loaded {len(raw):,} from {fname}")
        all_records.extend(process_records(raw, fname))

    print(f"\nTotal before dedup: {len(all_records):,}")
    all_records = deduplicate(all_records)
    print(f"Total after dedup:  {len(all_records):,}")

    max_samples = config["data"].get("max_samples", 300_000)

    if len(all_records) > max_samples:
        random.shuffle(all_records)
        all_records = all_records[:max_samples]
        print(f"Capped at {max_samples:,} samples")

    # Format into chat template
    formatted = [
        format_chat(r["problem"], r["solution"], r["answer"])
        for r in all_records
    ]

    # Train / eval split (95/5)
    split = int(len(formatted) * 0.95)
    train, eval_ = formatted[:split], formatted[split:]

    train_path = config["data"]["train_path"]
    eval_path = config["data"]["eval_path"]

    save_jsonl(train, train_path)
    save_jsonl(eval_, eval_path)

    print("\nData preparation complete.")