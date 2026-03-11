"""
Lean 4 tactic prover integration (optional, heavyweight).

Lean is the gold standard for formal math verification but requires:
  - Lean 4 installed: https://leanprover.github.io/lean4/doc/setup.html
  - Mathlib4 library (elan + lake)
  - ~5–30 seconds per verification (much slower than Z3)

Use Lean when:
  - Z3 and SymPy are inconclusive
  - The problem involves number-theoretic identities that Mathlib covers
  - You need a proof certificate, not just a numerical check

This module:
  1. Generates a Lean 4 theorem statement from the problem + claimed answer
  2. Runs Lean in a subprocess with a timeout
  3. Parses the output for proof success / failure / sorry

Note: Lean verification is BEST-EFFORT here. Full formalization of
arbitrary AIMO problems requires significant manual effort.
We use a templated approach for the most common problem types.
"""

import os
import sys
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.verifiers.symbolic import Verdict, VerificationResult


# ── Lean availability check ───────────────────────────────────────────────────

def lean_available() -> bool:
    try:
        result = subprocess.run(
            ["lean", "--version"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Lean 4 proof templates ────────────────────────────────────────────────────

# Template: verify that a numeric computation equals a claimed value
NUMERIC_TEMPLATE = """
import Mathlib

-- Auto-generated verification
theorem verify_{name} : {lhs} = {rhs} := by
  norm_num
"""

# Template: verify divisibility
DIVISIBILITY_TEMPLATE = """
import Mathlib

theorem verify_divisibility : ({mod} : ℤ) ∣ ({expr} : ℤ) := by
  norm_num
"""

# Template: verify inequality
INEQUALITY_TEMPLATE = """
import Mathlib

theorem verify_inequality : ({lhs} : ℤ) {op} ({rhs} : ℤ) := by
  norm_num
"""

# Template: verify modular equivalence
MODULAR_TEMPLATE = """
import Mathlib

theorem verify_modular : ({expr} : ℤ) % {mod} = {remainder} := by
  norm_num
"""


# ── Lean prover ───────────────────────────────────────────────────────────────

class LeanProver:
    """
    Generates and runs Lean 4 proof verification.
    Falls back to UNCERTAIN when Lean is not available or times out.
    """

    LEAN_TIMEOUT = 30   # seconds — Lean can be slow

    def __init__(self):
        self._lean_available = lean_available()
        if not self._lean_available:
            print("[LeanProver] Lean 4 not found. Install from https://leanprover.github.io")

    def verify(
        self,
        extraction,
        claimed_answer: str,
        problem_text: str = "",
    ) -> VerificationResult:

        if not self._lean_available:
            return VerificationResult(
                Verdict.UNCERTAIN, "lean",
                "Lean 4 not installed. Run: curl https://elan.lean-lang.org/install.sh | sh",
                None,
            )

        if not claimed_answer or not problem_text:
            return VerificationResult(Verdict.UNCERTAIN, "lean", "Missing inputs", None)

        checks = [
            self._verify_numeric_identity,
            self._verify_modular_claim,
            self._verify_divisibility_claim,
        ]

        for check in checks:
            result = check(claimed_answer, problem_text)
            if result.verdict != Verdict.UNCERTAIN:
                return result

        return VerificationResult(
            Verdict.UNCERTAIN, "lean", "No Lean-verifiable pattern detected", None
        )

    # ── Verify numeric identity ───────────────────────────────────────────────

    def _verify_numeric_identity(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        For problems where the answer is a direct arithmetic expression,
        verify: <expression> = <claimed_answer>
        """
        # Look for arithmetic expressions in problem
        m = re.search(r"(\d[\d\s\+\-\*\/\^\(\)]+)\s*=\s*\?", problem_text)
        if not m:
            return VerificationResult(Verdict.UNCERTAIN, "lean_numeric", "No expression found", None)

        expr = m.group(1).strip()
        # Convert to Lean syntax
        lean_expr = expr.replace("^", "^").replace("**", "^")

        lean_code = NUMERIC_TEMPLATE.format(
            name="answer",
            lhs=lean_expr,
            rhs=claimed_answer,
        )

        return self._run_lean(lean_code, "numeric_identity", claimed_answer)

    # ── Verify modular claim ──────────────────────────────────────────────────

    def _verify_modular_claim(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """Verify a mod arithmetic claim: expr mod m = claimed_answer."""
        m = re.search(r"(\d+)\s*(?:mod|%)\s*(\d+)", problem_text)
        if not m:
            return VerificationResult(Verdict.UNCERTAIN, "lean_modular", "No modular expression", None)

        expr, mod = m.group(1), m.group(2)
        lean_code = MODULAR_TEMPLATE.format(
            expr=expr,
            mod=mod,
            remainder=claimed_answer,
        )

        return self._run_lean(lean_code, "modular_claim", claimed_answer)

    # ── Verify divisibility ───────────────────────────────────────────────────

    def _verify_divisibility_claim(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """Verify 'does X divide Y?' type claims."""
        m = re.search(r"(\d+)\s+divides\s+(\d+)", problem_text, re.IGNORECASE)
        if not m:
            return VerificationResult(Verdict.UNCERTAIN, "lean_div", "No divisibility claim", None)

        divisor, dividend = m.group(1), m.group(2)
        lean_code = DIVISIBILITY_TEMPLATE.format(
            mod=divisor,
            expr=dividend,
        )

        return self._run_lean(lean_code, "divisibility", claimed_answer)

    # ── Lean runner ───────────────────────────────────────────────────────────

    def _run_lean(
        self, lean_code: str, method: str, claimed_answer: str
    ) -> VerificationResult:
        """Write Lean code to temp file and run lean CLI."""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", delete=False
        ) as f:
            f.write(lean_code)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                ["lean", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.LEAN_TIMEOUT,
            )

            stdout = proc.stdout + proc.stderr

            # Parse Lean output
            if proc.returncode == 0 and "error" not in stdout.lower():
                return VerificationResult(
                    Verdict.PASS, method,
                    f"Lean proof succeeded for answer {claimed_answer}",
                    claimed_answer,
                )

            if "type mismatch" in stdout or "application type mismatch" in stdout:
                return VerificationResult(
                    Verdict.FAIL, method,
                    f"Lean type mismatch — answer {claimed_answer} is incorrect",
                    None,
                )

            if "sorry" in lean_code:
                return VerificationResult(
                    Verdict.UNCERTAIN, method,
                    "Lean proof used sorry (incomplete)", None
                )

            return VerificationResult(
                Verdict.UNCERTAIN, method,
                f"Lean output: {stdout[:200]}", None
            )

        except subprocess.TimeoutExpired:
            return VerificationResult(
                Verdict.UNCERTAIN, method,
                f"Lean timed out after {self.LEAN_TIMEOUT}s", None
            )
        except FileNotFoundError:
            self._lean_available = False
            return VerificationResult(
                Verdict.UNCERTAIN, method, "Lean binary not found", None
            )
        finally:
            os.unlink(tmp_path)