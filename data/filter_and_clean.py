"""
Filters and cleans raw datasets for SFT training.

Rules:
  - Drop samples with fewer than 3 reasoning steps
  - Drop samples missing a final answer
  - Drop samples where SymPy can verify the answer is wrong
  - Deduplicate by problem text hash
  - Format into chat template for Qwen2.5-Math-7B-Instruct
  - 95/5 train/eval split
"""

import json 
import hashlib
import os
import re
from tqdm import tqdm
from utils.answer_utils import normalize_answer, sympy_verify

RAW_DIR = "./data/raw"
OUT_DIR = "./data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

SYSTEM_PROMPT = (
    "You are an expert math olympiad solver. "
    "Think step by step. "
    "At the end, state your final answer as a single integer on a new line prefixed with 'Answer:'."
)


# ── Formatting ────────────────────────────────────────────────────────────────

def format_chat(problem: str, solution: str, answer: str) -> dict:
    """Format a sample into Qwen2.5 chat template."""
    assistant_content = solution.strip()
    if answer and not assistant_content.endswith(answer):
        assistant_content += f"\n\nAnswer: {answer}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.strip()},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ── Filters ───────────────────────────────────────────────────────────────────

def has_enough_steps(solution: str, min_steps: int = 3) -> bool:
    """Check solution has at least min_steps reasoning steps."""
    # Count sentence-like reasoning units
    steps = re.split(r'\n+|(?<=[.!?])\s+', solution.strip())
    steps = [s for s in steps if len(s.strip()) > 10]
    return len(steps) >= min_steps


def has_valid_answer(answer: str) -> bool:
    """Answer must be non-empty and parseable."""
    if not answer or not answer.strip():
        return False
    norm = normalize_answer(answer)
    return norm is not None


def sympy_check_passes(problem: str, solution: str, answer: str) -> bool:
    """
    Optional: try to verify numeric answers with SymPy.
    Returns True if verification passes OR is inconclusive.
    Only returns False if verification actively detects a wrong answer.
    """
    try:
        result = sympy_verify(solution, answer)
        return result is not False  # None = inconclusive = keep
    except Exception:
        return True  # if verification errors, keep the sample


# ── Main pipeline ─────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def deduplicate(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for rec in records:
        h = hashlib.md5(rec["problem"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            out.append(rec)
    return out


def filter_records(records: list[dict], source: str) -> list[dict]:
    kept, dropped = [], 0
    for rec in tqdm(records, desc=f"Filtering {source}"):
        if not has_valid_answer(rec.get("answer", "")):
            dropped += 1
            continue
        if not has_enough_steps(rec.get("solution", "")):
            dropped += 1
            continue
        if not sympy_check_passes(rec["problem"], rec["solution"], rec["answer"]):
            dropped += 1
            continue
        kept.append(rec)
    print(f"  {source}: kept {len(kept):,}, dropped {dropped:,}")
    return kept


def save_jsonl(records: list[dict], path: str):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records):,} → {path}")


if __name__ == "__main__":
    all_records = []

    for fname in ["numina.jsonl", "math.jsonl", "aime.jsonl"]:
        path = f"{RAW_DIR}/{fname}"
        if not os.path.exists(path):
            print(f"Skipping {fname} (not found)")
            continue
        raw = load_jsonl(path)
        print(f"\nLoaded {len(raw):,} from {fname}")
        filtered = filter_records(raw, fname)
        all_records.extend(filtered)

    print(f"\nTotal before dedup: {len(all_records):,}")
    all_records = deduplicate(all_records)
    print(f"Total after dedup:  {len(all_records):,}")

    # Cap at 300K for single-GPU training
    if len(all_records) > 300_000:
        import random
        random.shuffle(all_records)
        all_records = all_records[:300_000]
        print(f"Capped at 300,000 samples")

    # Format into chat template
    formatted = [
        format_chat(r["problem"], r["solution"], r["answer"])
        for r in all_records
    ]

    # Train / eval split (95/5)
    split = int(len(formatted) * 0.95)
    train, eval_ = formatted[:split], formatted[split:]

    save_jsonl(train, f"{OUT_DIR}/sft_train.jsonl")
    save_jsonl(eval_,  f"{OUT_DIR}/sft_eval.jsonl")
    print("\nData preparation complete.")