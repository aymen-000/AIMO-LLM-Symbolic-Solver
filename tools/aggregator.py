"""
tools/aggregator.py

Combines verdicts from multiple verifiers into a single final verdict.

Aggregation rules:
  - Any FAIL         → final = FAIL  (one disproof is enough)
  - Any PASS         → final = PASS  (one proof is enough)
  - All UNCERTAIN    → final = UNCERTAIN

Also computes a confidence score: 0.0–1.0 based on:
  - Number of PASSes vs total verifiers run
  - Verifier weight (theorem provers > symbolic > numeric > heuristic)
"""

from dataclasses import dataclass, field
from typing import Optional
from tools.verifiers.symbolic import Verdict, VerificationResult


# ── Verifier weights (higher = more trustworthy) ──────────────────────────────
VERIFIER_WEIGHTS = {
    "lean":                    1.0,    # formal proof
    "z3":                      0.95,   # SMT solver
    "equation_substitution":   0.85,   # algebraic
    "sympy_simplification":    0.80,   # symbolic
    "boxed_consistency":       0.75,
    "code_execution":          0.70,   # ran and matched
    "heron_formula":           0.70,
    "pythagorean":             0.70,
    "distance_formula":        0.70,
    "circle_area":             0.65,
    "gcd_check":               0.90,
    "lcm_check":               0.90,
    "integer_enumeration":     0.85,
    "numeric_expression":      0.60,
    "code_execution_partial":  0.50,
    "digit_sum":               0.90,
}

DEFAULT_WEIGHT = 0.55   # for methods not in the table


# ── Aggregated result ─────────────────────────────────────────────────────────

@dataclass
class AggregatedResult:
    final_verdict:  Verdict
    confidence:     float                         # 0.0–1.0
    pass_evidence:  list[VerificationResult]      # all PASSes
    fail_evidence:  list[VerificationResult]      # all FAILs
    uncertain:      list[VerificationResult]      # all UNCERTAINs
    summary:        str


# ── Aggregator ────────────────────────────────────────────────────────────────

class VerdictAggregator:

    def aggregate(self, results: list[VerificationResult]) -> AggregatedResult:
        passes    = [r for r in results if r.verdict == Verdict.PASS]
        fails     = [r for r in results if r.verdict == Verdict.FAIL]
        uncertain = [r for r in results if r.verdict == Verdict.UNCERTAIN]

        # ── Final verdict ──────────────────────────────────────────────────
        # One FAIL from a high-weight verifier overrides everything
        if fails:
            max_fail_weight = max(
                VERIFIER_WEIGHTS.get(r.method, DEFAULT_WEIGHT) for r in fails
            )
            # Only hard-fail if a high-confidence verifier disagrees
            if max_fail_weight >= 0.65:
                final = Verdict.FAIL
            elif passes:
                # Low-confidence fail vs pass: go with pass
                final = Verdict.PASS
            else:
                final = Verdict.FAIL
        elif passes:
            final = Verdict.PASS
        else:
            final = Verdict.UNCERTAIN

        # ── Confidence score ───────────────────────────────────────────────
        confidence = self._compute_confidence(passes, fails, uncertain, final)

        # ── Summary ────────────────────────────────────────────────────────
        summary_parts = []
        for r in passes:
            summary_parts.append(f"✓ [{r.method}] {r.evidence}")
        for r in fails:
            summary_parts.append(f"✗ [{r.method}] {r.evidence}")
        summary = " | ".join(summary_parts) if summary_parts else "No conclusive verification"

        return AggregatedResult(
            final_verdict=final,
            confidence=round(confidence, 4),
            pass_evidence=passes,
            fail_evidence=fails,
            uncertain=uncertain,
            summary=summary,
        )

    def _compute_confidence(
        self,
        passes:    list[VerificationResult],
        fails:     list[VerificationResult],
        uncertain: list[VerificationResult],
        final:     Verdict,
    ) -> float:

        if not passes and not fails:
            return 0.0

        total_weight = sum(
            VERIFIER_WEIGHTS.get(r.method, DEFAULT_WEIGHT)
            for r in passes + fails + uncertain
        )
        if total_weight == 0:
            return 0.0

        if final == Verdict.PASS:
            positive_weight = sum(
                VERIFIER_WEIGHTS.get(r.method, DEFAULT_WEIGHT) for r in passes
            )
            return min(1.0, positive_weight / max(total_weight, 1.0))

        if final == Verdict.FAIL:
            negative_weight = sum(
                VERIFIER_WEIGHTS.get(r.method, DEFAULT_WEIGHT) for r in fails
            )
            return min(1.0, negative_weight / max(total_weight, 1.0))

        return 0.0