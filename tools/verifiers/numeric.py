"""
tools/verifiers/numeric.py

Numeric verification using numpy and scipy.

Strategy:
  - Extract Python code blocks from the solution
  - Run them in the sandboxed executor
  - Compare executor output to claimed answer
  - Also try evaluating math expressions numerically with tolerance

Useful for: integrals, numerical approximations, recursive sequences,
            problems where symbolic simplification is intractable.
"""

import os
import sys
import re
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.extractor import ExtractionResult
from tools.executor  import execute_code, execute_expression, ExecutionResult
from tools.verifiers.symbolic import Verdict, VerificationResult


NUMERIC_TOLERANCE = 1e-6


class NumericVerifier:
    """
    Executes code blocks from a solution and compares output to claimed answer.
    """

    def verify(
        self,
        extraction: ExtractionResult,
        claimed_answer: str,
        problem_text: str = "",
    ) -> VerificationResult:

        if not claimed_answer:
            return VerificationResult(Verdict.UNCERTAIN, "numeric", "No claimed answer", None)

        # Priority: execute full Python blocks, then evaluate expressions
        result = self._verify_code_blocks(extraction, claimed_answer)
        if result.verdict != Verdict.UNCERTAIN:
            return result

        result = self._verify_expressions(extraction, claimed_answer)
        if result.verdict != Verdict.UNCERTAIN:
            return result

        return VerificationResult(
            Verdict.UNCERTAIN, "numeric", "No executable content found", None
        )

    # ── Code block execution ──────────────────────────────────────────────────

    def _verify_code_blocks(
        self,
        extraction: ExtractionResult,
        claimed_answer: str,
    ) -> VerificationResult:
        """Run each Python code block and compare output to claimed answer."""

        runnable = [
            b for b in extraction.code_blocks
            if b.language in ("python", "sympy", "py", "sympy_inline", "unknown")
        ]

        if not runnable:
            return VerificationResult(Verdict.UNCERTAIN, "code_exec", "No runnable blocks", None)

        for block in runnable:
            exec_result = execute_code(block.code, timeout=15.0)

            if exec_result.timed_out:
                continue

            if not exec_result.success:
                # Try to extract any printed numbers from stdout before error
                nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", exec_result.output)
                if nums and self._answers_match(nums[-1], claimed_answer):
                    return VerificationResult(
                        Verdict.PASS,
                        "code_execution_partial",
                        f"Partial output matches: code printed {nums[-1]}",
                        nums[-1],
                    )
                continue

            computed = exec_result.answer or self._extract_last_number(exec_result.output)

            if computed is None:
                continue

            if self._answers_match(computed, claimed_answer):
                return VerificationResult(
                    Verdict.PASS,
                    "code_execution",
                    f"Executed code output {computed!r} matches claimed answer {claimed_answer!r}",
                    computed,
                )
            else:
                # Code ran successfully but disagrees — strong FAIL signal
                return VerificationResult(
                    Verdict.FAIL,
                    "code_execution",
                    f"Code computed {computed!r} but claimed answer is {claimed_answer!r}",
                    computed,
                )

        return VerificationResult(Verdict.UNCERTAIN, "code_exec", "All blocks inconclusive", None)

    # ── Expression evaluation ─────────────────────────────────────────────────

    def _verify_expressions(
        self,
        extraction: ExtractionResult,
        claimed_answer: str,
    ) -> VerificationResult:
        """Evaluate SymPy math expressions numerically."""

        for expr in extraction.math_expressions:
            sympy_str = expr.sympy_form.strip()
            # Only try closed-form numeric expressions (no free symbols)
            if re.search(r"\b[a-zA-Z]\b", sympy_str):
                continue  # has variables, skip
            if len(sympy_str) < 3:
                continue

            exec_result = execute_expression(sympy_str, timeout=5.0)
            if exec_result.success and exec_result.answer:
                computed = exec_result.answer
                if self._answers_match(computed, claimed_answer):
                    return VerificationResult(
                        Verdict.PASS,
                        "numeric_expression",
                        f"Expression {expr.raw!r} evaluates to {computed}",
                        computed,
                    )

        return VerificationResult(Verdict.UNCERTAIN, "numeric_expr", "No expressions matched", None)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _answers_match(self, computed: str, claimed: str) -> bool:
        """Compare two answer strings with numeric tolerance."""
        try:
            c1 = float(str(computed).strip().split()[0])
            c2 = float(str(claimed).strip().split()[0])
            return abs(c1 - c2) < NUMERIC_TOLERANCE
        except (ValueError, IndexError):
            # Fall back to string comparison
            return str(computed).strip() == str(claimed).strip()

    def _extract_last_number(self, text: str) -> Optional[str]:
        nums = re.findall(r"(?<!\w)(-?\d+(?:\.\d+)?)(?!\w)", text)
        return nums[-1] if nums else None