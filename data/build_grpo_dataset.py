"""
Selects hard problems for GRPO training.

Strategy:
  - Load AIME + MATH Level 4/5 problems
  - Run the SFT-merged model on each problem (greedy, N=8)
  - Keep only problems where solve rate is between 20%–80%
    (too easy = no gradient; too hard = no positive reward signal)
  - Save as grpo_hard.jsonl
"""

import json
import os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from utils.answer_utils import normalize_answer, extract_final_answer

MODEL_PATH = "./outputs/sft_merged"
RAW_DIR    = "./data/raw"
OUT_DIR    = "./data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES      = 8    
MIN_SOLVE_RATE = 0.20    # drop if model solves < 20% → too hard
MAX_SOLVE_RATE = 0.80    # drop if model solves > 80% → too easy
MAX_NEW_TOKENS = 1024
TEMPERATURE    = 0.8

SYSTEM_PROMPT = (
    "You are an expert math olympiad solver. "
    "Think step by step. "
    "At the end, state your final answer as a single integer on a new line prefixed with 'Answer:'."
)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_prompt(problem: str, tokenizer) -> str:
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": problem.strip()},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def estimate_solve_rate(
    problems: list[dict],
    llm: LLM,
    tokenizer,
) -> list[dict]:
    """
    For each problem, sample N_SAMPLES responses and compute solve rate.
    Returns problems annotated with solve_rate.
    """
    sampling_params = SamplingParams(
        n=N_SAMPLES,
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
        top_p=0.95,
    )

    prompts = [build_prompt(p["problem"], tokenizer) for p in problems]

    print(f"Running {N_SAMPLES} rollouts on {len(problems)} problems...")
    outputs = llm.generate(prompts, sampling_params)

    annotated = []
    for prob, output in zip(problems, outputs):
        gt = normalize_answer(prob["answer"])
        if gt is None:
            continue

        correct = 0
        for completion in output.outputs:
            pred = extract_final_answer(completion.text)
            pred_norm = normalize_answer(pred) if pred else None
            if pred_norm == gt:
                correct += 1

        solve_rate = correct / N_SAMPLES
        annotated.append({**prob, "solve_rate": solve_rate})

    return annotated


def main():
    # Load candidate problems (AIME + MATH hard)
    candidates = []

    aime_path = f"{RAW_DIR}/aime.jsonl"
    if os.path.exists(aime_path):
        aime = load_jsonl(aime_path)
        candidates.extend(aime)
        print(f"Loaded {len(aime):,} AIME problems")

    math_path = f"{RAW_DIR}/math.jsonl"
    if os.path.exists(math_path):
        math_all = load_jsonl(math_path)
        math_hard = [r for r in math_all if r.get("level", "") in ("Level 4", "Level 5")]
        candidates.extend(math_hard)
        print(f"Loaded {len(math_hard):,} MATH Level 4–5 problems")

    print(f"\nTotal candidates: {len(candidates):,}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    # Estimate solve rates
    annotated = estimate_solve_rate(candidates, llm, tokenizer)

    # Filter to calibrated difficulty
    hard = [
        r for r in annotated
        if MIN_SOLVE_RATE <= r["solve_rate"] <= MAX_SOLVE_RATE
    ]

    print(f"\nSolve-rate distribution:")
    print(f"  Too easy  (>{MAX_SOLVE_RATE:.0%}): {sum(1 for r in annotated if r['solve_rate'] > MAX_SOLVE_RATE):,}")
    print(f"  Calibrated:                       {len(hard):,}")
    print(f"  Too hard  (<{MIN_SOLVE_RATE:.0%}): {sum(1 for r in annotated if r['solve_rate'] < MIN_SOLVE_RATE):,}")

    # Save
    out_path = f"{OUT_DIR}/grpo_hard.jsonl"
    with open(out_path, "w") as f:
        for r in hard:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(hard):,} calibrated problems → {out_path}")


if __name__ == "__main__":
    main()