"""
AIMO Full Inference Pipeline
═══════════════════════════════════════════════════════════════════════════════

Stages (all in one run):
  1.  Load fine-tuned model into vLLM
  2.  Self-consistency sampling  (N paths per problem)
  3.  Tool execution + symbolic/numeric/Z3 verification per path
  4.  Voting  (majority | weighted | best_of_n)
  5.  Post-vote verification  (re-verify the chosen answer)
  6.  Fallback  (if chosen answer FAILs verification, try next candidate)
  7.  Output predictions with full audit trail

Usage
──────
  # Single problem (interactive)
  python inference.py --model ./outputs/grpo/final --problem "Find the sum..."

  # Batch from file
  python inference.py --model ./outputs/grpo/final \
                      --input  problems.jsonl \
                      --output predictions.jsonl \
                      --n 32 --strategy majority

  # Kaggle / competition submission mode
  python inference.py --model ./outputs/grpo/final \
                      --input  test.csv \
                      --output submission.csv \
                      --n 48 --strategy weighted --verify

Input formats accepted
──────────────────────
  .jsonl  —  one JSON per line, field "problem"
  .csv    —  columns "id", "problem"
  .txt    —  one problem per line

Output formats
──────────────
  .jsonl  —  full audit trail (answer + confidence + verification + paths)
  .csv    —  id, answer  (competition submission format)
"""

import os
import sys
import json
import csv
import time
import argparse
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from sampling.sampler   import SelfConsistencySampler, SamplingResult, SolutionPath
from sampling.voting    import MajorityVoter, WeightedVoter, BestOfNVoter
from sampling.scorers  import composite_score
from tools.extractor    import extract_from_response
from tools.router       import ProblemRouter, build_verifiers
from tools.aggregator   import VerdictAggregator
from tools.verifiers.symbolic import Verdict
from utils.answer_utils import normalize_answer, extract_final_answer
from utils.vram_utils   import print_vram

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aimo")


# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    # Model
    model_path:            str   = "./outputs/grpo/final"

    # Sampling
    n_samples:             int   = 32
    temperature:           float = 0.7
    top_p:                 float = 0.95
    max_new_tokens:        int   = 2048
    gpu_memory_utilization:float = 0.88

    # Voting
    strategy:              str   = "majority"   # majority | weighted | best_of_n

    # Verification
    verify:                bool  = True         # run tool verification
    max_verifiers:         int   = 4
    lean_enabled:          bool  = False
    early_stop_verify:     bool  = True         # stop on first confident verdict

    # Fallback
    fallback_enabled:      bool  = True         # try next-best answer if FAIL
    max_fallback_attempts: int   = 3

    # I/O
    input_path:            str   = ""
    output_path:           str   = "predictions.jsonl"
    batch_size:            int   = 8


@dataclass
class PathResult:
    """Single sampled solution path with verification attached."""
    response:        str
    answer:          Optional[str]
    tokens:          int
    score:           float          # composite heuristic score
    verdict:         str            # PASS / FAIL / UNCERTAIN / SKIPPED
    verdict_conf:    float
    verdict_summary: str


@dataclass
class Prediction:
    """Final output for one problem."""
    problem_id:      str
    problem:         str
    final_answer:    Optional[str]
    confidence:      float          # vote share of winning answer
    verdict:         str            # post-vote verification result
    verdict_conf:    float
    verdict_summary: str
    strategy:        str
    vote_counts:     dict           = field(default_factory=dict)
    n_total:         int            = 0
    n_valid:         int            = 0
    fallback_used:   bool           = False
    elapsed_sec:     float          = 0.0
    paths:           list[PathResult] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Voter factory
# ══════════════════════════════════════════════════════════════════════════════

def build_voter(strategy: str):
    if strategy == "majority":
        return MajorityVoter()
    if strategy == "weighted":
        return WeightedVoter(length_penalty=1e-5)
    if strategy == "best_of_n":
        return BestOfNVoter(score_fn=composite_score, aggregate_method="sum")
    raise ValueError(f"Unknown strategy: {strategy}")


# ══════════════════════════════════════════════════════════════════════════════
# Verification helper
# ══════════════════════════════════════════════════════════════════════════════

def verify_answer(
    problem:         str,
    answer:          str,
    solution_text:   str,
    router:          ProblemRouter,
    aggregator:      VerdictAggregator,
    config:          PipelineConfig,
) -> tuple[str, float, str]:
    """
    Run the tool verification stack for one (problem, answer, solution) triple.
    Returns (verdict_str, confidence, summary).
    """
    extraction     = extract_from_response(solution_text)
    routing        = router.route(problem, has_code=extraction.has_executable)
    verifier_names = [
        v for v in routing.verifier_names
        if config.lean_enabled or v != "lean"
    ][:config.max_verifiers]

    verifiers = build_verifiers(verifier_names)
    results   = []

    for verifier, name in zip(verifiers, verifier_names):
        try:
            res = verifier.verify(extraction, answer, problem)
            results.append(res)
            if config.early_stop_verify and res.verdict in (Verdict.PASS, Verdict.FAIL):
                from tools.aggregator import VERIFIER_WEIGHTS, DEFAULT_WEIGHT
                if VERIFIER_WEIGHTS.get(res.method, DEFAULT_WEIGHT) >= 0.70:
                    break
        except Exception as e:
            log.warning(f"Verifier {name} crashed: {e}")

    if not results:
        return Verdict.UNCERTAIN.value, 0.0, "No verifiers ran"

    agg = aggregator.aggregate(results)
    return agg.final_verdict.value, agg.confidence, agg.summary


# ══════════════════════════════════════════════════════════════════════════════
# Per-problem runner
# ══════════════════════════════════════════════════════════════════════════════

def run_one(
    problem_id:  str,
    problem:     str,
    sampler:     SelfConsistencySampler,
    voter,
    router:      ProblemRouter,
    aggregator:  VerdictAggregator,
    config:      PipelineConfig,
) -> Prediction:
    t0 = time.time()

    # ── Stage 1: Sample ───────────────────────────────────────────────────────
    log.info(f"[{problem_id}] Sampling {config.n_samples} paths…")
    sampling_result: SamplingResult = sampler.sample_one(problem)

    n_total = len(sampling_result.paths)
    n_valid = sampling_result.n_valid
    log.info(f"[{problem_id}] {n_valid}/{n_total} paths have valid answers")

    # ── Stage 2: Per-path verification (optional but informative) ─────────────
    path_results: list[PathResult] = []

    if config.verify:
        for path in sampling_result.paths:
            if path.answer is None:
                path_results.append(PathResult(
                    response=path.response, answer=None, tokens=path.tokens,
                    score=0.0, verdict="SKIPPED", verdict_conf=0.0, verdict_summary="No answer"
                ))
                continue

            # Find the best solution text for this path
            v, vc, vs = verify_answer(
                problem, path.answer, path.response,
                router, aggregator, config
            )
            sc = composite_score(path)
            path.score = sc
            path_results.append(PathResult(
                response=path.response,
                answer=path.answer,
                tokens=path.tokens,
                score=sc,
                verdict=v,
                verdict_conf=vc,
                verdict_summary=vs,
            ))

            # Promote verified paths: bump their score for best_of_n voter
            if v == Verdict.PASS.value:
                path.score = sc + 2.0    # strong boost to verified paths
            elif v == Verdict.FAIL.value:
                path.score = max(0.0, sc - 1.5)  # penalize failed paths
    else:
        for path in sampling_result.paths:
            sc = composite_score(path)
            path.score = sc
            path_results.append(PathResult(
                response=path.response, answer=path.answer,
                tokens=path.tokens, score=sc,
                verdict="SKIPPED", verdict_conf=0.0, verdict_summary="Verification disabled"
            ))

    # ── Stage 3: Vote ─────────────────────────────────────────────────────────
    final_answer = voter.aggregate(sampling_result)

    # Vote counts for audit
    from collections import Counter
    vote_counts = dict(Counter(
        normalize_answer(p.answer) or p.answer
        for p in sampling_result.paths if p.answer
    ))

    # Vote confidence = share of valid paths agreeing with winner
    if final_answer and n_valid > 0:
        norm_winner = normalize_answer(final_answer) or final_answer
        agreeing    = sum(
            1 for p in sampling_result.paths
            if p.answer and (normalize_answer(p.answer) or p.answer) == norm_winner
        )
        vote_conf = round(agreeing / n_valid, 4)
    else:
        vote_conf = 0.0

    log.info(f"[{problem_id}] Vote winner: {final_answer!r}  (conf={vote_conf:.1%})")

    # ── Stage 4: Post-vote verification ──────────────────────────────────────
    verdict, verdict_conf, verdict_summary = Verdict.UNCERTAIN.value, 0.0, ""
    fallback_used = False

    if config.verify and final_answer:
        # Use the best solution text for the winning answer
        best_solution = ""
        for path in sampling_result.paths:
            if path.answer and (normalize_answer(path.answer) or path.answer) == (
                normalize_answer(final_answer) or final_answer
            ):
                best_solution = path.response
                break

        verdict, verdict_conf, verdict_summary = verify_answer(
            problem, final_answer, best_solution, router, aggregator, config
        )
        log.info(f"[{problem_id}] Verification: {verdict} (conf={verdict_conf:.2f})")

        # ── Stage 5: Fallback if FAIL ─────────────────────────────────────────
        if verdict == Verdict.FAIL.value and config.fallback_enabled:
            log.warning(f"[{problem_id}] Primary answer FAILED verification, trying fallback…")

            # Rank candidate answers by vote count, excluding the failed one
            norm_failed  = normalize_answer(final_answer) or final_answer
            candidates   = sorted(
                [(ans, cnt) for ans, cnt in vote_counts.items() if ans != norm_failed],
                key=lambda x: x[1], reverse=True
            )

            for attempt, (cand_answer, cand_votes) in enumerate(
                candidates[:config.max_fallback_attempts]
            ):
                log.info(f"[{problem_id}] Fallback {attempt+1}: trying {cand_answer!r} ({cand_votes} votes)")
                # Find solution text for this candidate
                cand_solution = next(
                    (p.response for p in sampling_result.paths
                     if p.answer and (normalize_answer(p.answer) or p.answer) == cand_answer),
                    ""
                )
                fb_v, fb_vc, fb_vs = verify_answer(
                    problem, cand_answer, cand_solution, router, aggregator, config
                )
                if fb_v == Verdict.PASS.value:
                    log.info(f"[{problem_id}] Fallback answer {cand_answer!r} PASSED ✓")
                    final_answer  = cand_answer
                    verdict       = fb_v
                    verdict_conf  = fb_vc
                    verdict_summary = fb_vs
                    fallback_used = True

                    # Update vote confidence for new winner
                    if n_valid > 0:
                        agreeing = sum(
                            1 for p in sampling_result.paths
                            if p.answer and (normalize_answer(p.answer) or p.answer) == cand_answer
                        )
                        vote_conf = round(agreeing / n_valid, 4)
                    break

    elapsed = round(time.time() - t0, 2)
    log.info(f"[{problem_id}] Done in {elapsed}s  →  answer={final_answer!r}  verdict={verdict}")

    return Prediction(
        problem_id=problem_id,
        problem=problem,
        final_answer=final_answer,
        confidence=vote_conf,
        verdict=verdict,
        verdict_conf=verdict_conf,
        verdict_summary=verdict_summary,
        strategy=config.strategy,
        vote_counts=vote_counts,
        n_total=n_total,
        n_valid=n_valid,
        fallback_used=fallback_used,
        elapsed_sec=elapsed,
        paths=path_results,
    )


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_problems(path: str) -> list[tuple[str, str]]:
    """Returns list of (id, problem_text)."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        records = []
        with open(path) as f:
            for i, line in enumerate(f):
                if line.strip():
                    rec = json.loads(line)
                    records.append((
                        str(rec.get("id", i)),
                        rec["problem"],
                    ))
        return records

    if ext == ".csv":
        records = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                pid = row.get("id", str(i))
                problem = row.get("problem", row.get("Problem", ""))
                records.append((str(pid), problem))
        return records

    if ext == ".txt":
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        return [(str(i), line) for i, line in enumerate(lines)]

    raise ValueError(f"Unsupported input format: {ext}")


def save_predictions(predictions: list[Prediction], path: str):
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if ext == ".csv":
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "answer"])
            for pred in predictions:
                writer.writerow([pred.problem_id, pred.final_answer or ""])
        log.info(f"Saved CSV submission → {path}")
        return

    # Default: JSONL with full audit trail
    with open(path, "w") as f:
        for pred in predictions:
            # Convert dataclass to dict, keep paths as dicts too
            d = asdict(pred)
            f.write(json.dumps(d) + "\n")
    log.info(f"Saved JSONL predictions → {path}")


def print_summary(predictions: list[Prediction]):
    total     = len(predictions)
    answered  = sum(1 for p in predictions if p.final_answer is not None)
    passed    = sum(1 for p in predictions if p.verdict == "PASS")
    failed    = sum(1 for p in predictions if p.verdict == "FAIL")
    uncertain = sum(1 for p in predictions if p.verdict == "UNCERTAIN")
    fallbacks = sum(1 for p in predictions if p.fallback_used)
    avg_conf  = sum(p.confidence for p in predictions) / total
    avg_time  = sum(p.elapsed_sec for p in predictions) / total

    print("\n" + "═" * 58)
    print("  AIMO INFERENCE  —  SUMMARY")
    print("═" * 58)
    print(f"  Problems:          {total}")
    print(f"  Answered:          {answered} / {total}  ({answered/total:.1%})")
    print(f"  Avg vote conf:     {avg_conf:.1%}")
    print(f"  Verification:")
    print(f"    PASS:            {passed}  ({passed/total:.1%})")
    print(f"    FAIL:            {failed}  ({failed/total:.1%})")
    print(f"    UNCERTAIN:       {uncertain}  ({uncertain/total:.1%})")
    print(f"  Fallbacks used:    {fallbacks}")
    print(f"  Avg time/problem:  {avg_time:.1f}s")
    print("═" * 58)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AIMO Full Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    p.add_argument("--model",       default="./outputs/grpo/final",
                   help="Path to fine-tuned model checkpoint")
    # Input
    g = p.add_mutually_exclusive_group()
    g.add_argument("--problem", type=str,
                   help="Single problem string (interactive mode)")
    g.add_argument("--input",   type=str,
                   help="Input file (.jsonl / .csv / .txt)")
    # Output
    p.add_argument("--output",  default="predictions.jsonl",
                   help="Output file (.jsonl for full audit, .csv for submission)")
    # Sampling
    p.add_argument("--n",           type=int,   default=32,   help="Paths per problem")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p",       type=float, default=0.95)
    p.add_argument("--max_tokens",  type=int,   default=2048)
    # Voting
    p.add_argument("--strategy",    default="majority",
                   choices=["majority", "weighted", "best_of_n"])
    # Verification
    p.add_argument("--verify",      action="store_true", default=True,
                   help="Run tool verification (default: on)")
    p.add_argument("--no_verify",   action="store_true",
                   help="Disable all verification")
    p.add_argument("--lean",        action="store_true",
                   help="Enable Lean 4 theorem prover")
    p.add_argument("--max_verifiers", type=int, default=4)
    # Fallback
    p.add_argument("--no_fallback", action="store_true",
                   help="Disable fallback on verification failure")
    # Perf
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--gpu_util",    type=float, default=0.88)
    return p.parse_args()


def main():
    args = parse_args()

    config = PipelineConfig(
        model_path             = args.model,
        n_samples              = args.n,
        temperature            = args.temperature,
        top_p                  = args.top_p,
        max_new_tokens         = args.max_tokens,
        gpu_memory_utilization = args.gpu_util,
        strategy               = args.strategy,
        verify                 = args.verify and not args.no_verify,
        max_verifiers          = args.max_verifiers,
        lean_enabled           = args.lean,
        fallback_enabled       = not args.no_fallback,
        input_path             = args.input or "",
        output_path            = args.output,
        batch_size             = args.batch_size,
    )

    log.info("═" * 58)
    log.info("  AIMO INFERENCE PIPELINE")
    log.info("═" * 58)
    log.info(f"  Model:       {config.model_path}")
    log.info(f"  N samples:   {config.n_samples}")
    log.info(f"  Strategy:    {config.strategy}")
    log.info(f"  Verify:      {config.verify}")
    log.info(f"  Lean:        {config.lean_enabled}")
    log.info(f"  Fallback:    {config.fallback_enabled}")
    log.info("═" * 58)

    # ── Load problems ─────────────────────────────────────────────────────────
    if args.problem:
        problems = [("0", args.problem)]
    elif args.input:
        problems = load_problems(args.input)
        log.info(f"Loaded {len(problems)} problems from {args.input}")
    else:
        log.error("Provide --problem or --input")
        sys.exit(1)

    # ── Build components ──────────────────────────────────────────────────────
    print_vram("before model load")

    sampler = SelfConsistencySampler(
        model_path             = config.model_path,
        n_samples              = config.n_samples,
        temperature            = config.temperature,
        top_p                  = config.top_p,
        max_new_tokens         = config.max_new_tokens,
        gpu_memory_utilization = config.gpu_memory_utilization,
    )

    voter      = build_voter(config.strategy)
    router     = ProblemRouter()
    aggregator = VerdictAggregator()

    print_vram("after model load")

    # ── Run pipeline ──────────────────────────────────────────────────────────
    predictions: list[Prediction] = []

    for pid, problem in problems:
        try:
            pred = run_one(
                problem_id = pid,
                problem    = problem,
                sampler    = sampler,
                voter      = voter,
                router     = router,
                aggregator = aggregator,
                config     = config,
            )
            predictions.append(pred)

            # Live preview for single-problem mode
            if args.problem:
                print(f"\n{'═'*50}")
                print(f"  Answer:     {pred.final_answer}")
                print(f"  Confidence: {pred.confidence:.1%}")
                print(f"  Verdict:    {pred.verdict}  (conf={pred.verdict_conf:.2f})")
                print(f"  Votes:      {pred.vote_counts}")
                print(f"  Fallback:   {pred.fallback_used}")
                print(f"  Time:       {pred.elapsed_sec}s")
                print(f"{'═'*50}\n")

        except KeyboardInterrupt:
            log.warning("Interrupted — saving partial results…")
            break
        except Exception as e:
            log.error(f"[{pid}] Failed: {e}", exc_info=True)
            predictions.append(Prediction(
                problem_id=pid, problem=problem,
                final_answer=None, confidence=0.0,
                verdict=Verdict.UNCERTAIN.value, verdict_conf=0.0,
                verdict_summary=f"Pipeline error: {e}",
                strategy=config.strategy,
            ))

    # ── Save & summarise ──────────────────────────────────────────────────────
    if predictions:
        save_predictions(predictions, config.output_path)
        print_summary(predictions)


if __name__ == "__main__":
    main()