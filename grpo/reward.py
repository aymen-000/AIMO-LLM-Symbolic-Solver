"""
Reward function for GRPO training.

Reward design:
  1.0  — correct final answer (exact match after normalization)
  0.8  — correct answer but extracted from body, not "Answer:" line
  0.1  — wrong answer but response has valid CoT structure
  0.0  — wrong answer, no structure

"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.answer_utils import extract_final_answer, answers_match


# ── Structure checks ──────────────────────────────────────────────────────────

def has_cot_structure(text: str) -> bool:
    """
    Check that the response has a minimal CoT structure:
      - At least 3 reasoning sentences / lines
      - Contains some mathematical content (numbers, operators, =)
    """
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 10]
    has_math = bool(re.search(r"[\d+\-*/=^]", text))
    return len(lines) >= 3 and has_math


def has_answer_line(text: str) -> bool:
    """Check for explicit 'Answer: <value>' line."""
    return bool(re.search(r"Answer:\s*\S+", text, re.IGNORECASE))


# ── Reward function ───────────────────────────────────────────────────────────

def compute_reward(
    response: str,
    ground_truth: str,
    use_partial_rewards: bool = False,   # set True for sparse reward problems
) -> float:
    """
    Compute scalar reward for a single model response.

    Args:
        response:           Full model response text
        ground_truth:       Correct answer string
        use_partial_rewards: Whether to give partial credit for structure

    Returns:
        float reward in [0.0, 1.0]
    """
    pred = extract_final_answer(response)

    if pred is None:
        if use_partial_rewards and has_cot_structure(response):
            return 0.05  # at least it tried to reason
        return 0.0

    correct = answers_match(pred, ground_truth)

    if correct:
        # Full credit if answer is on explicit "Answer:" line
        if has_answer_line(response):
            return 1.0
        # Slight penalty if answer was inferred from body
        if use_partial_rewards:
            return 0.8
        return 1.0  # binary mode: still full credit

    # Wrong answer
    if use_partial_rewards and has_cot_structure(response):
        return 0.1

    return 0.0


def batch_rewards(
    responses: list[str],
    ground_truth: str,
    use_partial_rewards: bool = False,
) -> list[float]:
    """Compute rewards for a batch of responses to the same problem."""
    return [
        compute_reward(r, ground_truth, use_partial_rewards)
        for r in responses
    ]


# ── Advantage computation ─────────────────────────────────────────────────────

def compute_advantages(rewards: list[float]) -> list[float]:
    """
    Normalize rewards within a group to compute GRPO advantages.
    Â_i = (r_i - mean(r)) / (std(r) + eps)
    """
    import statistics
    if len(rewards) < 2:
        return [0.0] * len(rewards)

    mean = statistics.mean(rewards)
    std  = statistics.stdev(rewards) if len(rewards) > 1 else 1.0
    eps  = 1e-8

    return [(r - mean) / (std + eps) for r in rewards]