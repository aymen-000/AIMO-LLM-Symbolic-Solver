"""
tools/verifiers/combinatorial.py

Brute-force combinatorial verification.

Handles AIMO problem types that are amenable to exhaustive search:
  - Modular arithmetic (check answer mod m)
  - Integer constraints (find all integers satisfying conditions)
  - Counting problems (enumerate and count directly)
  - Divisibility / GCD / LCM conditions
  - Sum/product over ranges

These problems can often be independently verified in milliseconds
by brute-forcing the small integer search space.
"""

import os
import sys
import re
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.verifiers.symbolic import Verdict, VerificationResult
from tools.executor import execute_code

# ── Problem type detection ────────────────────────────────────────────────────

def _detect_modular(problem_text: str) -> Optional[int]:
    """Return modulus if problem asks for answer mod m."""
    m = re.search(
        r"(?:find|compute|what is).*?(\d+)(?:\s*\)|\s*$).*?(?:mod(?:ulo)?|remainder)",
        problem_text, re.IGNORECASE
    )
    if m:
        return int(m.group(1))
    m = re.search(r"mod(?:ulo)?\s*(\d{3,})", problem_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _detect_integer_range(problem_text: str) -> Optional[tuple[int, int]]:
    """Try to detect the search range for integer problems."""
    # "positive integers less than N"
    m = re.search(r"(?:positive )?integers?\s+(?:less than|up to|at most|not exceeding)\s+(\d+)", problem_text, re.IGNORECASE)
    if m:
        return (1, int(m.group(1)))
    # "integers from A to B"
    m = re.search(r"integers?\s+from\s+(\d+)\s+to\s+(\d+)", problem_text, re.IGNORECASE)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


# ── Verifier ──────────────────────────────────────────────────────────────────

class CombinatorialVerifier:
    """
    Generates and runs brute-force Python verification scripts
    based on the structure of the problem text.
    """

    MAX_BRUTE_FORCE = 100_000   # cap iteration to keep runtime < 5s

    def verify(
        self,
        extraction,
        claimed_answer: str,
        problem_text: str = "",
    ) -> VerificationResult:

        if not claimed_answer or not problem_text:
            return VerificationResult(Verdict.UNCERTAIN, "combinatorial", "Missing inputs", None)

        checks = [
            self._check_modular_arithmetic,
            self._check_integer_enumeration,
            self._check_divisibility,
            self._check_digit_sum,
        ]

        for check in checks:
            result = check(claimed_answer, problem_text)
            if result.verdict != Verdict.UNCERTAIN:
                return result

        return VerificationResult(
            Verdict.UNCERTAIN, "combinatorial", "No combinatorial pattern matched", None
        )

    # ── Check 1: Modular arithmetic ───────────────────────────────────────────

    def _check_modular_arithmetic(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        mod = _detect_modular(problem_text)
        if mod is None:
            return VerificationResult(Verdict.UNCERTAIN, "modular", "No modulus detected", None)

        try:
            ans_int = int(claimed_answer)
        except ValueError:
            return VerificationResult(Verdict.UNCERTAIN, "modular", "Non-integer answer", None)

        # Verify answer is in valid range [0, mod-1]
        if not (0 <= ans_int < mod):
            return VerificationResult(
                Verdict.FAIL, "modular",
                f"Answer {ans_int} is outside valid range [0, {mod-1}]",
                str(ans_int),
            )

        # We can't brute-force the modular check without knowing the full expression,
        # but we can at least confirm range validity
        return VerificationResult(
            Verdict.UNCERTAIN, "modular",
            f"Answer {ans_int} is in valid range mod {mod}, but full check requires expression",
            None,
        )

    # ── Check 2: Integer enumeration ─────────────────────────────────────────

    def _check_integer_enumeration(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        Build a brute-force enumeration script from the problem description.
        Works for 'how many integers satisfy...' type problems.
        """
        range_info = _detect_integer_range(problem_text)
        if range_info is None:
            return VerificationResult(Verdict.UNCERTAIN, "enumeration", "No range detected", None)

        lo, hi = range_info
        if hi - lo > self.MAX_BRUTE_FORCE:
            return VerificationResult(
                Verdict.UNCERTAIN, "enumeration",
                f"Range [{lo}, {hi}] too large for brute force", None
            )

        # Look for simple divisibility conditions in problem
        divides = re.findall(r"divisible by (\d+)", problem_text, re.IGNORECASE)
        not_divides = re.findall(r"not divisible by (\d+)", problem_text, re.IGNORECASE)

        if not divides and not not_divides:
            return VerificationResult(Verdict.UNCERTAIN, "enumeration", "No conditions found", None)

        # Build verification code
        conditions = []
        for d in divides:
            conditions.append(f"n % {d} == 0")
        for d in not_divides:
            conditions.append(f"n % {d} != 0")

        cond_str = " and ".join(conditions)
        code = f"""
count = sum(1 for n in range({lo}, {hi + 1}) if {cond_str})
result = count
print(f"Brute force count: {{count}}")
"""
        exec_result = execute_code(code, timeout=10.0)

        if exec_result.success and exec_result.answer:
            computed = exec_result.answer.strip()
            try:
                if int(computed) == int(claimed_answer):
                    return VerificationResult(
                        Verdict.PASS, "integer_enumeration",
                        f"Brute force count {computed} matches claimed answer",
                        computed,
                    )
                else:
                    return VerificationResult(
                        Verdict.FAIL, "integer_enumeration",
                        f"Brute force count {computed} ≠ claimed answer {claimed_answer}",
                        computed,
                    )
            except ValueError:
                pass

        return VerificationResult(Verdict.UNCERTAIN, "enumeration", "Enumeration inconclusive", None)

    # ── Check 3: Divisibility / GCD / LCM ────────────────────────────────────

    def _check_divisibility(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """Check GCD/LCM problems by direct computation."""
        # GCD pattern: "gcd(a, b)" or "greatest common divisor of A and B"
        m = re.search(
            r"(?:gcd|greatest common (?:divisor|factor))\s*(?:of\s+)?(\d+)\s+and\s+(\d+)",
            problem_text, re.IGNORECASE
        )
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            code = f"from math import gcd\nresult = gcd({a}, {b})"
            exec_result = execute_code(code, timeout=3.0)
            if exec_result.success and exec_result.answer:
                computed = exec_result.answer.strip()
                verdict = Verdict.PASS if computed == claimed_answer.strip() else Verdict.FAIL
                return VerificationResult(
                    verdict, "gcd_check",
                    f"GCD({a},{b}) = {computed}, claimed = {claimed_answer}",
                    computed,
                )

        # LCM pattern
        m = re.search(
            r"(?:lcm|least common multiple)\s*(?:of\s+)?(\d+)\s+and\s+(\d+)",
            problem_text, re.IGNORECASE
        )
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            code = f"from math import gcd\nresult = ({a} * {b}) // gcd({a}, {b})"
            exec_result = execute_code(code, timeout=3.0)
            if exec_result.success and exec_result.answer:
                computed = exec_result.answer.strip()
                verdict = Verdict.PASS if computed == claimed_answer.strip() else Verdict.FAIL
                return VerificationResult(
                    verdict, "lcm_check",
                    f"LCM({a},{b}) = {computed}, claimed = {claimed_answer}",
                    computed,
                )

        return VerificationResult(Verdict.UNCERTAIN, "divisibility", "No GCD/LCM pattern", None)

    # ── Check 4: Digit sum ────────────────────────────────────────────────────

    def _check_digit_sum(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """Verify digit sum / digit product claims."""
        m = re.search(r"sum of (?:the )?digits of (\d+)", problem_text, re.IGNORECASE)
        if m:
            n = m.group(1)
            code = f"result = sum(int(d) for d in '{n}')"
            exec_result = execute_code(code, timeout=2.0)
            if exec_result.success and exec_result.answer:
                computed = exec_result.answer.strip()
                verdict = Verdict.PASS if computed == claimed_answer.strip() else Verdict.FAIL
                return VerificationResult(
                    verdict, "digit_sum",
                    f"digit_sum({n}) = {computed}, claimed = {claimed_answer}",
                    computed,
                )

        return VerificationResult(Verdict.UNCERTAIN, "digit_sum", "No digit pattern", None)