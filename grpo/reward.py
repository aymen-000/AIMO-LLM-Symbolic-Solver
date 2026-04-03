"""
Reward Function for GRPO 

Features:
- Smooth reward (clamped [0,1])
- Self-consistency bonus
- Intermediate math/process reward
- Strong CoT structure detection
- Length regularization
- Confidence penalty
- Fake reasoning penalty
- Advantage normalization (clipped)
"""

import re
import statistics
from collections import Counter
from utils.answer_utils import extract_final_answer, answers_match

# ── Structure checks ──────────────────────────────────────────────────────────

def has_weak_cot(text: str) -> bool:
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
    has_math = bool(re.search(r"[\d+\-*/=^]", text))
    return len(lines) >= 3 and has_math

def has_strong_cot(text: str) -> bool:
    steps = re.findall(r"(Step\s*\d+:)", text, re.IGNORECASE)
    equations = re.findall(r"=", text)
    return len(steps) >= 2 and len(equations) >= 2

def has_answer_line(text: str) -> bool:
    return bool(re.search(r"Answer:\s*\S+", text, re.IGNORECASE))

# ── Process-level rewards ─────────────────────────────────────────────────────

def intermediate_math_score(text: str) -> float:
    equations = re.findall(r"\d+\s*[\+\-\*/]\s*\d+\s*=\s*\d+", text)
    return min(len(equations) * 0.02, 0.1)

def confidence_score(text: str) -> float:
    if re.search(r"(maybe|probably|I think|unsure)", text, re.IGNORECASE):
        return -0.1
    return 0.05

def length_penalty(text: str, target_len: int = 200) -> float:
    length = len(text.split())
    return -abs(length - target_len) / target_len

def fake_reasoning_penalty(text: str) -> float:
    if "Step 1" in text and not re.search(r"\d", text):
        return -0.2
    return 0.0

# ── Self-consistency bonus ────────────────────────────────────────────────────

def self_consistency_bonus(responses: list[str]) -> list[float]:
    preds = [extract_final_answer(r) for r in responses]
    counts = Counter(preds)
    total = len(preds)
    bonuses = []
    for p in preds:
        bonuses.append(counts[p] / total if p is not None else 0.0)
    return bonuses

# ── Core reward ───────────────────────────────────────────────────────────────

def compute_reward(
    response: str,
    ground_truth: str,
    use_partial_rewards: bool = True,
) -> float:
    reward = 0.0
    pred = extract_final_answer(response)

    # Correctness
    if pred is not None and answers_match(pred, ground_truth):
        reward += 0.7

    # Format rewards
    if has_answer_line(response):
        reward += 0.1
    if has_strong_cot(response):
        reward += 0.1
    elif use_partial_rewards and has_weak_cot(response):
        reward += 0.05

    # Process rewards
    reward += intermediate_math_score(response)

    # Regularization
    reward += confidence_score(response)
    reward += 0.05 * length_penalty(response)
    reward += fake_reasoning_penalty(response)

    return max(0.0, min(1.0, reward))

# ── Batch reward with consistency ─────────────────────────────────────────────

def batch_rewards(
    responses: list[str],
    ground_truth: str,
    use_partial_rewards: bool = True,
) -> list[float]:
    base_rewards = [compute_reward(r, ground_truth, use_partial_rewards) for r in responses]
    bonuses = self_consistency_bonus(responses)
    final_rewards = [min(1.0, r + 0.2 * b) for r, b in zip(base_rewards, bonuses)]
    return final_rewards

# ── Advantage computation ─────────────────────────────────────────────────────

def compute_advantages(rewards: list[float]) -> list[float]:
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean = statistics.mean(rewards)
    std = statistics.stdev(rewards) + 1e-8
    advantages = [(r - mean) / std for r in rewards]
    advantages = [max(min(a, 5.0), -5.0) for a in advantages]
    return advantages