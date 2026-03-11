"""
Aggregation strategies to pick the final answer from N sampled paths.

Three strategies in order of complexity:
  1. MajorityVoter       — simple frequency count
  2. WeightedVoter       — frequency + length penalty
  3. BestOfNVoter        — uses a scoring function (e.g. ORM score)
"""

import os
import sys
from collections import Counter, defaultdict
from typing import Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.answer_utils import normalize_answer
from sampling.sampler import SamplingResult, SolutionPath


# ── Helpers ───────────────────────────────────────────────────────────────────

def group_by_answer(
    paths: list[SolutionPath],
) -> dict[str, list[SolutionPath]]:
    """Group paths by their normalized answer."""
    groups: dict[str, list[SolutionPath]] = defaultdict(list)
    for path in paths:
        if path.answer is None:
            continue
        key = normalize_answer(path.answer) or path.answer
        groups[key].append(path)
    return dict(groups)


# ── Strategy 1: Majority Vote ─────────────────────────────────────────────────

class MajorityVoter:
    """
    Pick the answer that appears most frequently across N paths.
    Ties are broken by the answer that appears first.
    """

    def aggregate(self, result: SamplingResult) -> Optional[str]:
        groups = group_by_answer(result.paths)
        if not groups:
            return None

        best_answer = max(groups, key=lambda a: len(groups[a]))
        return best_answer

    def vote_counts(self, result: SamplingResult) -> dict[str, int]:
        groups = group_by_answer(result.paths)
        return {a: len(paths) for a, paths in groups.items()}


# ── Strategy 2: Weighted Vote ─────────────────────────────────────────────────

class WeightedVoter:
    """
    Weighted majority vote.

    Each path contributes a weight = 1 / (1 + length_penalty * token_count).
    This downweights very long (potentially rambling) solutions.
    Paths without a valid answer get weight 0.

    Args:
        length_penalty: weight decay per token (default: very small)
    """

    def __init__(self, length_penalty: float = 1e-5):
        self.length_penalty = length_penalty

    def _path_weight(self, path: SolutionPath) -> float:
        if path.answer is None:
            return 0.0
        # Penalise extremely long responses
        return 1.0 / (1.0 + self.length_penalty * path.tokens)

    def aggregate(self, result: SamplingResult) -> Optional[str]:
        groups  = group_by_answer(result.paths)
        if not groups:
            return None

        # Sum weights per answer group
        scores: dict[str, float] = {}
        for answer, paths in groups.items():
            scores[answer] = sum(self._path_weight(p) for p in paths)

        return max(scores, key=scores.__getitem__)

    def weighted_scores(self, result: SamplingResult) -> dict[str, float]:
        groups = group_by_answer(result.paths)
        return {
            a: sum(self._path_weight(p) for p in paths)
            for a, paths in groups.items()
        }


# ── Strategy 3: Best-of-N ─────────────────────────────────────────────────────

class BestOfNVoter:
    """
    Best-of-N selection using an external scoring function.

    The score_fn receives a SolutionPath and returns a float.
    Typical use: pass in an ORM (Outcome Reward Model) or PRM score.

    For a lightweight alternative, pass in a heuristic scorer
    (see sampling/scorers.py).

    Args:
        score_fn: callable(SolutionPath) -> float
        aggregate_method: how to combine scores across paths with same answer
            "max"  — take the single best path per answer group, then vote
            "mean" — average scores per answer group, then pick highest
            "sum"  — sum scores per answer group (equivalent to weighted vote)
    """

    def __init__(
        self,
        score_fn: Callable[[SolutionPath], float],
        aggregate_method: str = "sum",
    ):
        assert aggregate_method in ("max", "mean", "sum")
        self.score_fn = score_fn
        self.aggregate_method = aggregate_method

    def _group_score(self, paths: list[SolutionPath]) -> float:
        scores = [self.score_fn(p) for p in paths]
        if self.aggregate_method == "max":
            return max(scores)
        if self.aggregate_method == "mean":
            return sum(scores) / len(scores)
        return sum(scores)  # "sum"

    def aggregate(self, result: SamplingResult) -> Optional[str]:
        groups = group_by_answer(result.paths)
        if not groups:
            return None

        group_scores = {a: self._group_score(paths) for a, paths in groups.items()}
        return max(group_scores, key=group_scores.__getitem__)

    def best_path(self, result: SamplingResult) -> Optional[SolutionPath]:
        """Return the single highest-scored solution path."""
        valid = [p for p in result.paths if p.answer is not None]
        if not valid:
            return None
        return max(valid, key=self.score_fn)

    def scored_paths(self, result: SamplingResult) -> list[tuple[float, SolutionPath]]:
        """Return all paths sorted by score descending."""
        valid = [p for p in result.paths if p.answer is not None]
        scored = [(self.score_fn(p), p) for p in valid]
        return sorted(scored, key=lambda x: x[0], reverse=True)