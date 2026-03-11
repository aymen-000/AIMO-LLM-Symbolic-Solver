"""
Symbolic verification using SymPy.

Checks:
  1. Can claimed answer satisfy the equations found in the solution?
  2. Does symbolic simplification of the claimed answer match the solution's expressions?
  3. Can we solve the problem's equation symbolically and compare?

Verdict: PASS / FAIL / UNCERTAIN
"""

import re
import os
import sys
from dataclasses import dataclass
from typing import Optional
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.extractor import ExtractionResult

try:
    from sympy import (
        sympify, simplify, solve, expand, factor, Symbol, symbols,
        Eq, Rational, Integer, N as sympy_N, latex,
        parse_expr, SympifyError
    )
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# ── Verdict ───────────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    PASS      = "PASS"
    FAIL      = "FAIL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class VerificationResult:
    verdict:   Verdict
    method:    str             # which check produced the verdict
    evidence:  str             # human-readable explanation
    computed:  Optional[str]   # what we computed (if any)


# ── Symbolic verifier ─────────────────────────────────────────────────────────

class SymbolicVerifier:
    """
    Uses SymPy to verify a claimed answer against the math in a solution.
    """

    def verify(
        self,
        extraction: ExtractionResult,
        claimed_answer: str,
        problem_text: str = "",
    ) -> VerificationResult:

        if not SYMPY_AVAILABLE:
            return VerificationResult(
                Verdict.UNCERTAIN, "sympy_unavailable",
                "SymPy not installed", None
            )

        if claimed_answer is None:
            return VerificationResult(
                Verdict.UNCERTAIN, "no_answer",
                "No claimed answer to verify", None
            )

        # Try checks in priority order
        checks = [
            self._check_equation_substitution,
            self._check_sympy_expressions,
            self._check_boxed_consistency,
        ]

        for check in checks:
            result = check(extraction, claimed_answer, problem_text)
            if result.verdict != Verdict.UNCERTAIN:
                return result

        return VerificationResult(
            Verdict.UNCERTAIN, "no_verifiable_math",
            "Could not find verifiable mathematical structure", None
        )

    # ── Check 1: Equation substitution ───────────────────────────────────────

    def _check_equation_substitution(
        self,
        extraction: ExtractionResult,
        claimed_answer: str,
        problem_text: str,
    ) -> VerificationResult:
        """
        Find equations in the solution and substitute the claimed answer.
        If LHS - RHS simplifies to 0, the answer satisfies the equation.
        """
        # Collect equation strings from math expressions
        eq_strings = []
        for expr in extraction.math_expressions:
            s = expr.sympy_form
            if "=" in s and "==" not in s:
                eq_strings.append(s)

        if not eq_strings:
            return VerificationResult(Verdict.UNCERTAIN, "eq_sub", "No equations found", None)

        try:
            ans_val = sympify(claimed_answer)
        except SympifyError:
            return VerificationResult(Verdict.UNCERTAIN, "eq_sub", "Can't parse answer", None)

        # Try to identify the variable (x, n, k, a, b, m are common)
        candidate_vars = ["n", "x", "k", "a", "b", "m", "r", "t"]

        for eq_str in eq_strings[:5]:  # check up to 5 equations
            try:
                parts = eq_str.split("=", 1)
                if len(parts) != 2:
                    continue
                lhs = sympify(parts[0].strip())
                rhs = sympify(parts[1].strip())
                diff = simplify(lhs - rhs)

                # Substitute claimed answer for each candidate variable
                for var_name in candidate_vars:
                    var = Symbol(var_name)
                    if var in diff.free_symbols:
                        substituted = diff.subs(var, ans_val)
                        result_val  = simplify(substituted)
                        if result_val == 0:
                            return VerificationResult(
                                Verdict.PASS,
                                "equation_substitution",
                                f"Answer {claimed_answer} satisfies equation: {eq_str}",
                                str(result_val),
                            )
                        elif result_val.is_number and abs(float(sympy_N(result_val))) > 1e-6:
                            return VerificationResult(
                                Verdict.FAIL,
                                "equation_substitution",
                                f"Answer {claimed_answer} does NOT satisfy: {eq_str} "
                                f"(residual = {result_val})",
                                str(result_val),
                            )
            except Exception:
                continue

        return VerificationResult(Verdict.UNCERTAIN, "eq_sub", "Substitution inconclusive", None)

    # ── Check 2: SymPy expression simplification ──────────────────────────────

    def _check_sympy_expressions(
        self,
        extraction: ExtractionResult,
        claimed_answer: str,
        problem_text: str,
    ) -> VerificationResult:
        """
        Try to simplify each math expression and compare to claimed answer.
        If an expression simplifies to the claimed answer, that's a PASS.
        """
        try:
            ans_val = sympify(claimed_answer)
        except SympifyError:
            return VerificationResult(Verdict.UNCERTAIN, "simplify", "Can't parse answer", None)

        for expr in extraction.math_expressions:
            try:
                sym_expr = sympify(expr.sympy_form)
                # Only try to evaluate fully numeric expressions
                if not sym_expr.free_symbols:
                    evaluated = simplify(sym_expr)
                    diff = simplify(evaluated - ans_val)
                    if diff == 0 or (diff.is_number and abs(float(sympy_N(diff))) < 1e-9):
                        return VerificationResult(
                            Verdict.PASS,
                            "sympy_simplification",
                            f"Expression {expr.raw} simplifies to {claimed_answer}",
                            str(evaluated),
                        )
            except Exception:
                continue

        return VerificationResult(Verdict.UNCERTAIN, "simplify", "No numeric expression matched", None)

    # ── Check 3: Boxed answer consistency ────────────────────────────────────

    def _check_boxed_consistency(
        self,
        extraction: ExtractionResult,
        claimed_answer: str,
        problem_text: str,
    ) -> VerificationResult:
        """
        If multiple \boxed{} values appear in the solution, check they're consistent.
        Inconsistent boxes = FAIL.
        """
        boxed_values = []
        for expr in extraction.math_expressions:
            m = re.search(r"\\boxed\{([^}]+)\}", expr.raw)
            if m:
                boxed_values.append(m.group(1).strip())

        if len(boxed_values) < 2:
            return VerificationResult(Verdict.UNCERTAIN, "boxed_consistency", "Not enough boxed values", None)

        try:
            first = sympify(boxed_values[0])
            for bv in boxed_values[1:]:
                other = sympify(bv)
                if simplify(first - other) != 0:
                    return VerificationResult(
                        Verdict.FAIL,
                        "boxed_consistency",
                        f"Inconsistent boxed answers: {boxed_values}",
                        str(boxed_values),
                    )
            # All consistent — check against claimed answer
            if simplify(first - sympify(claimed_answer)) == 0:
                return VerificationResult(
                    Verdict.PASS,
                    "boxed_consistency",
                    f"All boxed values agree with claimed answer {claimed_answer}",
                    str(first),
                )
        except Exception:
            pass

        return VerificationResult(Verdict.UNCERTAIN, "boxed_consistency", "Boxed check inconclusive", None)