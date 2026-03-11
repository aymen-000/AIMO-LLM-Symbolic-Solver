"""

End-to-end tool execution and verification pipeline.

Flow per SolutionPath:
  1. Extract code blocks + math expressions (extractor.py)
  2. Route to verifiers based on problem type (router.py)
  3. Run each verifier in order, stopping early on PASS or FAIL
  4. Aggregate verdicts (aggregator.py)
  5. Return VerifiedPrediction

CLI usage:
  python tools/pipeline.py \
    --predictions outputs/predictions.jsonl \
    --output      outputs/verified.jsonl
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.extractor  import extract_from_response
from tools.router     import ProblemRouter, build_verifiers
from tools.aggregator import VerdictAggregator, AggregatedResult
from tools.verifiers.symbolic import Verdict


# ── Output structure ──────────────────────────────────────────────────────────

@dataclass
class VerifiedPrediction:
    problem:         str
    claimed_answer:  Optional[str]
    final_verdict:   str              # PASS / FAIL / UNCERTAIN
    confidence:      float
    summary:         str
    verifiers_run:   list[str]
    domains:         list[str]
    # Carry-through from sampling
    vote_confidence: float = 0.0
    vote_counts:     dict  = None

    def __post_init__(self):
        if self.vote_counts is None:
            self.vote_counts = {}


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ToolVerificationPipeline:
    """
    Runs tool extraction + multi-verifier checking on sampled predictions.
    """

    def __init__(
        self,
        early_stop: bool = True,          # stop on first conclusive verdict
        max_verifiers: int = 4,           # cap verifiers per problem
        lean_enabled: bool = False,       # Lean is slow; off by default
    ):
        self.early_stop    = early_stop
        self.max_verifiers = max_verifiers
        self.lean_enabled  = lean_enabled
        self.router        = ProblemRouter()
        self.aggregator    = VerdictAggregator()

    def verify_one(
        self,
        problem:         str,
        claimed_answer:  Optional[str],
        solution_text:   str = "",
        vote_confidence: float = 0.0,
        vote_counts:     dict  = None,
    ) -> VerifiedPrediction:

        if not claimed_answer:
            return VerifiedPrediction(
                problem=problem,
                claimed_answer=None,
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                summary="No answer to verify",
                verifiers_run=[],
                domains=[],
                vote_confidence=vote_confidence,
                vote_counts=vote_counts or {},
            )

        # Step 1: Extract
        extraction = extract_from_response(solution_text)
        has_code   = extraction.has_executable

        # Step 2: Route
        routing = self.router.route(problem, has_code=has_code)
        verifier_names = [
            v for v in routing.verifier_names
            if self.lean_enabled or v != "lean"
        ][:self.max_verifiers]

        # Step 3: Build verifiers
        verifiers = build_verifiers(verifier_names)

        # Step 4: Run verifiers
        all_results = []
        verifiers_run = []

        for verifier, name in zip(verifiers, verifier_names):
            try:
                result = verifier.verify(extraction, claimed_answer, problem)
                all_results.append(result)
                verifiers_run.append(name)

                # Early stop on conclusive result from trusted verifier
                if self.early_stop and result.verdict in (Verdict.PASS, Verdict.FAIL):
                    from tools.aggregator import VERIFIER_WEIGHTS, DEFAULT_WEIGHT
                    w = VERIFIER_WEIGHTS.get(result.method, DEFAULT_WEIGHT)
                    if w >= 0.70:
                        break  # confident enough, skip remaining verifiers
            except Exception as e:
                print(f"  [pipeline] Verifier {name} crashed: {e}")
                continue

        # Step 5: Aggregate
        if all_results:
            agg = self.aggregator.aggregate(all_results)
        else:
            from tools.aggregator import AggregatedResult
            agg = AggregatedResult(
                final_verdict=Verdict.UNCERTAIN,
                confidence=0.0,
                pass_evidence=[],
                fail_evidence=[],
                uncertain=[],
                summary="No verifiers produced results",
            )

        return VerifiedPrediction(
            problem=problem,
            claimed_answer=claimed_answer,
            final_verdict=agg.final_verdict.value,
            confidence=agg.confidence,
            summary=agg.summary,
            verifiers_run=verifiers_run,
            domains=[d.value for d in routing.domains],
            vote_confidence=vote_confidence,
            vote_counts=vote_counts or {},
        )

    def verify_batch(
        self,
        predictions: list[dict],
    ) -> list[VerifiedPrediction]:
        """
        Verify a batch of prediction dicts.
        Each dict should have: problem, final_answer, (optional) solution_text,
        confidence, vote_counts.
        """
        results = []
        for pred in tqdm(predictions, desc="Verifying"):
            # Use first valid solution path text if available
            solution_text = pred.get("solution_text", "")
            if not solution_text and "paths" in pred:
                # Try to find a solution path matching the claimed answer
                for path in pred["paths"]:
                    if path.get("answer") == pred.get("final_answer"):
                        solution_text = path.get("response", "")
                        break

            verified = self.verify_one(
                problem=pred["problem"],
                claimed_answer=pred.get("final_answer"),
                solution_text=solution_text,
                vote_confidence=pred.get("confidence", 0.0),
                vote_counts=pred.get("vote_counts", {}),
            )
            results.append(verified)
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Tool verification pipeline")
    parser.add_argument("--predictions", required=True, help="Input predictions JSONL")
    parser.add_argument("--output",      required=True, help="Output verified JSONL")
    parser.add_argument("--lean",        action="store_true", help="Enable Lean 4 prover")
    parser.add_argument("--max_verifiers", type=int, default=4)
    parser.add_argument("--no_early_stop", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.predictions) as f:
        predictions = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    pipeline = ToolVerificationPipeline(
        early_stop=not args.no_early_stop,
        max_verifiers=args.max_verifiers,
        lean_enabled=args.lean,
    )

    verified = pipeline.verify_batch(predictions)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for v in verified:
            f.write(json.dumps(asdict(v)) + "\n")

    # Summary
    pass_count      = sum(1 for v in verified if v.final_verdict == "PASS")
    fail_count      = sum(1 for v in verified if v.final_verdict == "FAIL")
    uncertain_count = sum(1 for v in verified if v.final_verdict == "UNCERTAIN")
    avg_conf        = sum(v.confidence for v in verified) / len(verified)

    print(f"\n── Verification Summary ─────────────────────────────")
    print(f"  PASS:      {pass_count:>4}  ({pass_count/len(verified):.1%})")
    print(f"  FAIL:      {fail_count:>4}  ({fail_count/len(verified):.1%})")
    print(f"  UNCERTAIN: {uncertain_count:>4}  ({uncertain_count/len(verified):.1%})")
    print(f"  Avg conf:  {avg_conf:.3f}")
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()