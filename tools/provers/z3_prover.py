"""
tools/provers/z3_prover.py

Z3 SMT solver integration for constraint-based verification.

Z3 is ideal for AIMO problems involving:
  - Integer constraints (find all integers where f(n) = k)
  - Modular arithmetic proofs (prove a ≡ b mod m)
  - Inequality systems (verify bounds)
  - Diophantine equations (prove existence / uniqueness)
  - Logical constraints over finite domains

Install: pip install z3-solver

Usage model:
  We translate the problem's constraints into Z3 assertions,
  then use the solver to:
    VERIFY  — prove the claimed answer satisfies all constraints
    REFUTE  — find a counterexample that contradicts the answer
    UNIQUE  — prove no other answer satisfies the constraints
"""

import os
import sys
import re
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tools.verifiers.symbolic import Verdict, VerificationResult
from tools.executor import execute_code

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False


# ── Z3 Prover ─────────────────────────────────────────────────────────────────

class Z3Prover:
    """
    Uses Z3 SMT solver to verify integer/modular constraint problems.

    Two modes:
      1. Direct:   Run pre-built Z3 scripts for detected problem patterns
      2. Indirect: Run Z3 verification code generated inside the sandbox
    """

    TIMEOUT_MS = 10_000   # Z3 solver timeout in milliseconds

    def verify(
        self,
        extraction,
        claimed_answer: str,
        problem_text: str = "",
    ) -> VerificationResult:

        if not Z3_AVAILABLE:
            return VerificationResult(
                Verdict.UNCERTAIN, "z3", "z3-solver not installed (pip install z3-solver)", None
            )

        if not claimed_answer or not problem_text:
            return VerificationResult(Verdict.UNCERTAIN, "z3", "Missing inputs", None)

        checks = [
            self._check_modular_constraint,
            self._check_diophantine,
            self._check_inequality_bound,
            self._check_uniqueness,
        ]

        for check in checks:
            result = check(claimed_answer, problem_text)
            if result.verdict != Verdict.UNCERTAIN:
                return result

        return VerificationResult(
            Verdict.UNCERTAIN, "z3", "No Z3-verifiable pattern detected", None
        )

    # ── Check 1: Modular constraints ──────────────────────────────────────────

    def _check_modular_constraint(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        Verify answers to 'find x such that x ≡ a (mod m)' problems.
        Also handles systems of congruences (Chinese Remainder Theorem style).
        """
        # Extract congruences: "x ≡ a (mod m)" or "x mod m = a"
        congruences = re.findall(
            r"(?:x|n|k)\s*(?:≡|\\equiv|mod\s*\w*\s*=|≡)\s*(\d+)\s*\(?\s*(?:mod|modulo)\s*(\d+)\s*\)?",
            problem_text, re.IGNORECASE
        )
        if not congruences:
            congruences_alt = re.findall(
                r"remainder\s+(?:of\s+)?(?:x|n)\s+(?:divided\s+by|mod)\s+(\d+)\s+is\s+(\d+)",
                problem_text, re.IGNORECASE
            )
            if congruences_alt:
                congruences = [(b, m) for m, b in congruences_alt]

        if not congruences:
            return VerificationResult(Verdict.UNCERTAIN, "z3_modular", "No congruences found", None)

        # Build Z3 verification via sandbox
        z3_code = self._build_modular_z3_code(congruences, claimed_answer)
        exec_result = execute_code(z3_code, timeout=15.0)

        return self._parse_z3_output(exec_result, "modular_constraint", claimed_answer)

    def _build_modular_z3_code(
        self, congruences: list[tuple[str, str]], claimed_answer: str
    ) -> str:
        constraints = []
        for remainder, modulus in congruences:
            constraints.append(f"    solver.add(x % {modulus} == {remainder})")

        constraints_str = "\n".join(constraints)
        return f"""
import z3

solver = z3.Solver()
solver.set("timeout", {self.TIMEOUT_MS})
x = z3.Int("x")

# Domain: non-negative integers up to reasonable bound
solver.add(x >= 0)
solver.add(x <= 10**7)

# Congruence constraints
{constraints_str}

# First: check if claimed answer satisfies all constraints
solver.push()
solver.add(x == {claimed_answer})
sat1 = solver.check()
solver.pop()

if str(sat1) == "sat":
    # Now check uniqueness: is there another solution in range?
    solver.push()
    solver.add(x != {claimed_answer})
    sat2 = solver.check()
    solver.pop()
    if str(sat2) == "unsat":
        result = "UNIQUE_PASS"
        print(f"Z3: Answer {claimed_answer} uniquely satisfies all congruences")
    else:
        model = solver.model()
        other = model.eval(x)
        result = f"PASS_NOT_UNIQUE: another solution exists: {{other}}"
        print(f"Z3: Answer {claimed_answer} satisfies constraints but so does {{other}}")
else:
    result = "FAIL"
    print(f"Z3: Answer {claimed_answer} does NOT satisfy congruences")
"""

    # ── Check 2: Diophantine equations ────────────────────────────────────────

    def _check_diophantine(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        Verify simple Diophantine equations like ax + by = c over integers.
        """
        # Pattern: "integer solutions to ax + by = c"
        m = re.search(
            r"(\d+)\s*x\s*\+\s*(\d+)\s*y\s*=\s*(\d+)",
            problem_text, re.IGNORECASE
        )
        if not m:
            return VerificationResult(Verdict.UNCERTAIN, "z3_diophantine", "No Diophantine pattern", None)

        a, b, c = m.group(1), m.group(2), m.group(3)

        # Build Z3 code to count solutions in a bounded domain
        z3_code = f"""
import z3

solver = z3.Solver()
solver.set("timeout", {self.TIMEOUT_MS})
x, y = z3.Ints("x y")

solver.add({a}*x + {b}*y == {c})
solver.add(x >= 0, y >= 0)     # adjust bounds based on problem

count = 0
solutions = []
while solver.check() == z3.sat and count < 1000:
    model = solver.model()
    xv, yv = model.eval(x).as_long(), model.eval(y).as_long()
    solutions.append((xv, yv))
    solver.add(z3.Or(x != xv, y != yv))
    count += 1

result = count
print(f"Z3: Found {{count}} non-negative integer solutions")
print(f"Solutions: {{solutions[:5]}}")
"""
        exec_result = execute_code(z3_code, timeout=15.0)
        return self._parse_z3_output(exec_result, "diophantine", claimed_answer)

    # ── Check 3: Inequality bounds ────────────────────────────────────────────

    def _check_inequality_bound(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        Verify 'maximum/minimum value' claims using Z3 optimization.
        """
        is_max = bool(re.search(r"maximum|largest|greatest|max", problem_text, re.IGNORECASE))
        is_min = bool(re.search(r"minimum|smallest|least|min", problem_text, re.IGNORECASE))

        if not (is_max or is_min):
            return VerificationResult(Verdict.UNCERTAIN, "z3_bound", "Not a max/min problem", None)

        # Extract simple inequality constraints from problem
        constraints = re.findall(
            r"(\w)\s*([<>]=?)\s*(\d+)",
            problem_text
        )
        if not constraints:
            return VerificationResult(Verdict.UNCERTAIN, "z3_bound", "No constraints extracted", None)

        # Build Z3 optimization code
        var_names = list(set(c[0] for c in constraints if c[0].isalpha() and c[0] != 'e'))
        if not var_names:
            return VerificationResult(Verdict.UNCERTAIN, "z3_bound", "No variables", None)

        var_decls  = ", ".join(f"{v}" for v in var_names)
        int_decls  = "\n".join(f'{v} = z3.Int("{v}")' for v in var_names)
        cond_parts = []
        for var, op, val in constraints:
            if var in var_names:
                cond_parts.append(f"opt.add({var} {op} {val})")
        cond_str = "\n".join(cond_parts)

        opt_goal = var_names[0]
        opt_fn   = "maximize" if is_max else "minimize"

        z3_code = f"""
import z3

opt = z3.Optimize()
opt.set("timeout", {self.TIMEOUT_MS})
{int_decls}

{cond_str}

h = opt.{opt_fn}({opt_goal})
status = opt.check()
if str(status) == "sat":
    result = str(opt.model().eval({opt_goal}))
    print(f"Z3 optimal value: {{result}}")
else:
    result = "INFEASIBLE"
    print("Z3: problem is infeasible")
"""
        exec_result = execute_code(z3_code, timeout=15.0)
        return self._parse_z3_output(exec_result, "inequality_bound", claimed_answer)

    # ── Check 4: Uniqueness verification ─────────────────────────────────────

    def _check_uniqueness(
        self, claimed_answer: str, problem_text: str
    ) -> VerificationResult:
        """
        If the problem asks for a unique value, verify there's exactly one solution.
        Runs from extracted code blocks if they contain Z3 setup.
        """
        # Check if any code block already uses z3
        return VerificationResult(Verdict.UNCERTAIN, "z3_uniqueness", "Generic uniqueness not implemented", None)

    # ── Output parser ─────────────────────────────────────────────────────────

    def _parse_z3_output(
        self,
        exec_result,
        method: str,
        claimed_answer: str,
    ) -> VerificationResult:

        if exec_result.timed_out:
            return VerificationResult(Verdict.UNCERTAIN, method, "Z3 timed out", None)

        if not exec_result.success:
            return VerificationResult(Verdict.UNCERTAIN, method, f"Z3 error: {exec_result.error}", None)

        output   = exec_result.output
        computed = exec_result.answer

        if "UNIQUE_PASS" in (computed or "") or "UNIQUE_PASS" in output:
            return VerificationResult(
                Verdict.PASS, method,
                f"Z3: answer {claimed_answer} uniquely satisfies constraints",
                claimed_answer,
            )

        if "PASS" in (computed or "") or "satisfies" in output.lower():
            return VerificationResult(
                Verdict.PASS, method,
                f"Z3: answer {claimed_answer} satisfies constraints. {output[:200]}",
                computed,
            )

        if "FAIL" in (computed or "") or "does not" in output.lower():
            return VerificationResult(
                Verdict.FAIL, method,
                f"Z3: answer {claimed_answer} violates constraints. {output[:200]}",
                computed,
            )

        # Try numeric comparison
        if computed:
            try:
                if abs(float(computed) - float(claimed_answer)) < 1e-6:
                    return VerificationResult(
                        Verdict.PASS, method,
                        f"Z3 computed {computed} matches claimed {claimed_answer}",
                        computed,
                    )
                else:
                    return VerificationResult(
                        Verdict.FAIL, method,
                        f"Z3 computed {computed} ≠ claimed {claimed_answer}",
                        computed,
                    )
            except ValueError:
                pass

        return VerificationResult(Verdict.UNCERTAIN, method, f"Z3 output: {output[:100]}", computed)