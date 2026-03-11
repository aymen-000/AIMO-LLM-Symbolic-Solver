"""
Lightweight heuristic scoring functions for Best-of-N selection.
Use these when you don't have a trained ORM/PRM.

Each scorer takes a SolutionPath and returns a float.
Higher = better.

Scorers:
  length_normalized_score  — penalizes very short or very long responses
  step_count_score         — rewards more reasoning steps
  confidence_score         — rewards consistent answers within the path
  composite_score          — weighted combination of all heuristics
"""

import re
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sampling.sampler import SolutionPath


# ── Individual scorers ────────────────────────────────────────────────────────

def length_normalized_score(path: SolutionPath) -> float:
    """
    Penalize paths that are too short (likely incomplete) or
    too long (likely repetitive/rambling).

    Sweet spot: 200–800 tokens.
    """
    t = path.tokens
    if t < 50:
        return 0.0
    if t <= 200:
        return t / 200.0                  # ramp up
    if t <= 800:
        return 1.0                        # ideal range
    # Soft decay beyond 800 tokens
    return max(0.1, 1.0 - (t - 800) / 3000.0)


def step_count_score(path: SolutionPath) -> float:
    """
    Reward responses with more distinct reasoning steps.
    Proxy: count numbered steps, "therefore", "so", "thus", newlines.
    """
    text = path.response
    step_markers = len(re.findall(
        r"(?:\n\s*\d+[\.\)]\s|therefore|thus|so,|hence|we get|this gives)",
        text, re.IGNORECASE
    ))
    equation_lines = len(re.findall(r"\n.*=.*\n", text))
    total = step_markers + equation_lines
    # Normalize: 5+ steps → full score
    return min(1.0, total / 5.0)


def confidence_score(path: SolutionPath) -> float:
    """
    Check whether the answer appears consistently throughout the response.
    If the model states the same answer in multiple places, it's more confident.
    """
    if not path.answer:
        return 0.0

    answer_str = str(path.answer).strip()
    # Count occurrences of the answer value in the text
    occurrences = path.response.count(answer_str)

    if occurrences == 0:
        return 0.0
    if occurrences == 1:
        return 0.5   # only at the end, no internal confirmation
    return min(1.0, 0.5 + 0.1 * (occurrences - 1))


def has_answer_line_score(path: SolutionPath) -> float:
    """Binary: 1.0 if response has explicit 'Answer: X' line, else 0.5."""
    if re.search(r"Answer:\s*\S+", path.response, re.IGNORECASE):
        return 1.0
    return 0.5


def latex_usage_score(path: SolutionPath) -> float:
    """
    Responses with LaTeX math formatting tend to be more structured.
    Reward moderate LaTeX usage.
    """
    latex_count = len(re.findall(r"\$[^$]+\$|\\[a-zA-Z]+\{", path.response))
    return min(1.0, latex_count / 10.0)


# ── Composite scorer ──────────────────────────────────────────────────────────

def composite_score(path: SolutionPath) -> float:
    """
    Weighted combination of all heuristic scorers.

    Weights reflect empirical importance for AIMO-style problems.
    Tune these based on your validation set performance.
    """
    weights = {
        "length":     0.25,
        "steps":      0.30,
        "confidence": 0.25,
        "answer_fmt": 0.15,
        "latex":      0.05,
    }

    scores = {
        "length":     length_normalized_score(path),
        "steps":      step_count_score(path),
        "confidence": confidence_score(path),
        "answer_fmt": has_answer_line_score(path),
        "latex":      latex_usage_score(path),
    }

    return sum(weights[k] * scores[k] for k in weights)