"""
Problem type classifier → verifier assignment.

Classifies a problem into one or more math domains and returns
an ordered list of verifiers to run (most likely to succeed first).

Domains:
  algebraic      → SymbolicVerifier, NumericVerifier
  numeric        → NumericVerifier, SymbolicVerifier
  combinatorial  → CombinatorialVerifier, Z3Prover
  geometric      → GeometricVerifier, NumericVerifier
  number_theory  → Z3Prover, CombinatorialVerifier, SymbolicVerifier
  has_code       → NumericVerifier (always first if code present)
  
  
i will use LLM to do this 
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Domain(str, Enum):
    ALGEBRAIC     = "algebraic"
    NUMERIC       = "numeric"
    COMBINATORIAL = "combinatorial"
    GEOMETRIC     = "geometric"
    NUMBER_THEORY = "number_theory"
    UNKNOWN       = "unknown"


@dataclass
class RoutingDecision:
    domains:          list[Domain]
    verifier_names:   list[str]     # ordered: first = highest priority
    reasoning:        str


# ── Keyword maps ──────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS = {
    Domain.ALGEBRAIC: [
        "solve", "equation", "polynomial", "roots", "factor", "expand",
        "simplify", "expression", "algebraic", "variable", "quadratic",
        "linear", "system of equations", "simultaneous",
    ],
    Domain.COMBINATORIAL: [
        "how many", "count", "number of", "ways", "combinations", "permutations",
        "arrangements", "choose", "select", "committee", "path", "sequence",
        "divisible", "multiples", "integers", "remainder", "modulo", "mod",
        "sum of digits", "digit", "last digit",
    ],
    Domain.GEOMETRIC: [
        "triangle", "circle", "square", "rectangle", "polygon", "angle",
        "area", "perimeter", "distance", "radius", "diameter", "chord",
        "tangent", "parallel", "perpendicular", "coordinate", "point",
        "line", "segment", "hypotenuse", "right angle",
    ],
    Domain.NUMBER_THEORY: [
        "prime", "gcd", "lcm", "divisor", "factor", "congruent", "mod",
        "euler", "fermat", "coprime", "relatively prime", "perfect number",
        "fibonacci", "arithmetic progression", "geometric progression",
    ],
    Domain.NUMERIC: [
        "compute", "evaluate", "calculate", "find the value", "what is",
        "sum", "product", "series", "sequence", "limit", "integral",
    ],
}

# Verifier priority per domain
DOMAIN_VERIFIERS = {
    Domain.ALGEBRAIC:     ["symbolic", "numeric", "z3"],
    Domain.COMBINATORIAL: ["combinatorial", "z3", "numeric"],
    Domain.GEOMETRIC:     ["geometric", "numeric", "symbolic"],
    Domain.NUMBER_THEORY: ["z3", "combinatorial", "symbolic"],
    Domain.NUMERIC:       ["numeric", "symbolic"],
    Domain.UNKNOWN:       ["numeric", "symbolic", "combinatorial"],
}


# ── Router ────────────────────────────────────────────────────────────────────

class ProblemRouter:
    """
    Classifies a problem and returns the optimal verifier ordering.
    """

    def route(self, problem_text: str, has_code: bool = False) -> RoutingDecision:
        text_lower = problem_text.lower()

        # Score each domain
        scores: dict[Domain, int] = {d: 0 for d in Domain}
        matched_keywords: dict[Domain, list[str]] = {d: [] for d in Domain}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    scores[domain] += 1
                    matched_keywords[domain].append(kw)

        # Get top domains (score > 0), sorted by score
        active_domains = [
            d for d in sorted(scores, key=scores.__getitem__, reverse=True)
            if scores[d] > 0
        ]

        if not active_domains:
            active_domains = [Domain.UNKNOWN]

        # Build verifier list: deduplicated, preserving priority order
        seen      = set()
        verifiers = []

        # If code is present, numeric executor always goes first
        if has_code:
            verifiers.append("numeric")
            seen.add("numeric")

        # Add verifiers for each active domain
        for domain in active_domains[:2]:  # use top 2 domains
            for v in DOMAIN_VERIFIERS.get(domain, []):
                if v not in seen:
                    verifiers.append(v)
                    seen.add(v)

        # Always include lean last as heavyweight fallback
        if "lean" not in seen:
            verifiers.append("lean")

        # Build reasoning string
        top = active_domains[0] if active_domains else Domain.UNKNOWN
        kws = matched_keywords.get(top, [])[:3]
        reasoning = (
            f"Primary domain: {top.value} "
            f"(matched: {', '.join(kws) if kws else 'none'}). "
            f"Verifier order: {' → '.join(verifiers)}"
        )

        return RoutingDecision(
            domains=active_domains,
            verifier_names=verifiers,
            reasoning=reasoning,
        )


# ── Verifier factory ──────────────────────────────────────────────────────────

def build_verifiers(names: list[str]) -> list[Any]:
    """
    Instantiate verifier objects from a list of names.
    Lazy imports to avoid loading unused dependencies.
    """
    verifiers = []
    for name in names:
        try:
            if name == "symbolic":
                from tools.verifiers.symbolic import SymbolicVerifier
                verifiers.append(SymbolicVerifier())
            elif name == "numeric":
                from tools.verifiers.numeric import NumericVerifier
                verifiers.append(NumericVerifier())
            elif name == "combinatorial":
                from tools.verifiers.combinatorial import CombinatorialVerifier
                verifiers.append(CombinatorialVerifier())
            elif name == "geometric":
                from tools.verifiers.geometric import GeometricVerifier
                verifiers.append(GeometricVerifier())
            elif name == "z3":
                from tools.provers.z3_prover import Z3Prover
                verifiers.append(Z3Prover())
        except ImportError as e:
            print(f"[router] Could not load verifier '{name}': {e}")
    return verifiers