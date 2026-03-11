"""
End-to-end self-consistency pipeline.

Takes a list of problems, runs the sampler, applies a voting strategy,
and returns final answers with confidence metadata.

Usage:
    python sampling/pipeline.py \
        --model   ./outputs/grpo/final \
        --input   problems.jsonl \
        --output  predictions.jsonl \
        --n       32 \
        --strategy majority
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from sampling.sampler   import SelfConsistencySampler, SamplingResult
from sampling.voting    import MajorityVoter, WeightedVoter, BestOfNVoter
from sampling.scorers   import composite_score
from utils.answer_utils import normalize_answer


# ── Output structure ──────────────────────────────────────────────────────────

@dataclass
class Prediction:
    problem:         str
    final_answer:    Optional[str]    # chosen answer
    confidence:      float            # vote share of chosen answer
    n_total:         int              # total paths sampled
    n_valid:         int              # paths with extractable answer
    vote_counts:     dict             # {answer: count} for majority voter
    strategy:        str


# ── Pipeline ──────────────────────────────────────────────────────────────────

class SelfConsistencyPipeline:
    """
    Wraps sampler + voter into a single callable pipeline.

    Args:
        model_path: fine-tuned model checkpoint
        n_samples:  number of paths to sample per problem
        strategy:   "majority" | "weighted" | "best_of_n"
        temperature: sampling temperature
        max_new_tokens: max tokens per path
    """

    def __init__(
        self,
        model_path: str,
        n_samples: int = 32,
        strategy: str = "majority",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ):
        self.n_samples = n_samples
        self.strategy  = strategy

        self.sampler = SelfConsistencySampler(
            model_path=model_path,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )

        # Build voter
        if strategy == "majority":
            self.voter = MajorityVoter()
        elif strategy == "weighted":
            self.voter = WeightedVoter(length_penalty=1e-5)
        elif strategy == "best_of_n":
            self.voter = BestOfNVoter(
                score_fn=composite_score,
                aggregate_method="sum",
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Choose: majority | weighted | best_of_n")

    def _confidence(self, result: SamplingResult, final_answer: Optional[str]) -> float:
        """Fraction of valid paths that agree with the chosen answer."""
        if final_answer is None or result.n_valid == 0:
            return 0.0
        norm = normalize_answer(final_answer) or final_answer
        matching = sum(
            1 for p in result.paths
            if p.answer and (normalize_answer(p.answer) or p.answer) == norm
        )
        return round(matching / result.n_valid, 4)

    def _vote_counts(self, result: SamplingResult) -> dict:
        if hasattr(self.voter, "vote_counts"):
            return self.voter.vote_counts(result)
        from collections import Counter
        answers = [
            normalize_answer(p.answer) or p.answer
            for p in result.paths if p.answer
        ]
        return dict(Counter(answers))

    def predict_one(self, problem: str) -> Prediction:
        result       = self.sampler.sample_one(problem)
        final_answer = self.voter.aggregate(result)
        return Prediction(
            problem=problem,
            final_answer=final_answer,
            confidence=self._confidence(result, final_answer),
            n_total=len(result.paths),
            n_valid=result.n_valid,
            vote_counts=self._vote_counts(result),
            strategy=self.strategy,
        )

    def predict_batch(self, problems: list[str], batch_size: int = 8) -> list[Prediction]:
        """
        Run pipeline on multiple problems.
        Processes in batches for vLLM throughput efficiency.
        """
        predictions = []
        for i in tqdm(range(0, len(problems), batch_size), desc="Sampling"):
            batch   = problems[i : i + batch_size]
            results = self.sampler.sample_batch(batch)
            for result in results:
                final_answer = self.voter.aggregate(result)
                predictions.append(Prediction(
                    problem=result.problem,
                    final_answer=final_answer,
                    confidence=self._confidence(result, final_answer),
                    n_total=len(result.paths),
                    n_valid=result.n_valid,
                    vote_counts=self._vote_counts(result),
                    strategy=self.strategy,
                ))
        return predictions


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Self-consistency sampling pipeline")
    parser.add_argument("--model",    required=True,  help="Path to fine-tuned model")
    parser.add_argument("--input",    required=True,  help="Input JSONL (field: 'problem')")
    parser.add_argument("--output",   required=True,  help="Output JSONL path")
    parser.add_argument("--n",        type=int, default=32, help="Paths per problem (default 32)")
    parser.add_argument("--strategy", default="majority",
                        choices=["majority", "weighted", "best_of_n"],
                        help="Voting strategy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load problems
    with open(args.input) as f:
        records = [json.loads(l) for l in f if l.strip()]
    problems = [r["problem"] for r in records]
    print(f"Loaded {len(problems)} problems from {args.input}")

    # Build pipeline
    pipeline = SelfConsistencyPipeline(
        model_path=args.model,
        n_samples=args.n,
        strategy=args.strategy,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    # Run
    predictions = pipeline.predict_batch(problems, batch_size=args.batch_size)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for pred in predictions:
            f.write(json.dumps(asdict(pred)) + "\n")

    # Summary stats
    answered  = sum(1 for p in predictions if p.final_answer is not None)
    avg_conf  = sum(p.confidence for p in predictions) / len(predictions)
    avg_valid = sum(p.n_valid for p in predictions) / len(predictions)

    print(f"\n── Results ──────────────────────────────────")
    print(f"  Problems:          {len(predictions)}")
    print(f"  Answered:          {answered} / {len(predictions)}")
    print(f"  Avg confidence:    {avg_conf:.1%}")
    print(f"  Avg valid paths:   {avg_valid:.1f} / {args.n}")
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()