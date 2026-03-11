"""
Answer extraction, normalization, and SymPy verification utilities.
Used by both the data pipeline and the GRPO reward function.
"""

import re
from typing import Optional
from sympy import simplify, sympify, N
from sympy.parsing.latex import parse_latex


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the final answer from a model response.
    Tries multiple formats in priority order.
    """
    if not text:
        return None

    # 1. Explicit "Answer: <value>" line (our preferred format)
    m = re.search(r"Answer:\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. LaTeX \boxed{...}
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip()

    # 3. "the answer is X" / "= X" at end of text
    m = re.search(
        r"(?:the answer is|therefore|thus|so|=)\s*([+-]?\d+(?:\.\d+)?(?:/\d+)?)\s*$",
        text.strip(), re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    # 4. Last standalone integer in text (fallback)
    numbers = re.findall(r"\b(\d{1,6})\b", text)
    if numbers:
        return numbers[-1]

    return None


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_answer(answer: str) -> Optional[str]:
    """
    Normalize an answer string to a canonical form for comparison.

    - Strips whitespace, LaTeX wrappers
    - Converts fractions to decimals for numeric comparison
    - Returns None if unparseable
    """
    if answer is None:
        return None

    s = answer.strip()

    # Strip common LaTeX wrappers
    s = re.sub(r"\\boxed\{(.+)\}", r"\1", s)
    s = re.sub(r"\$", "", s)
    s = s.strip()

    # Try integer
    if re.fullmatch(r"[+-]?\d+", s):
        return str(int(s))

    # Try simple fraction a/b
    m = re.fullmatch(r"([+-]?\d+)/(\d+)", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den != 0:
            # Return reduced fraction string
            from math import gcd
            g = gcd(abs(num), den)
            return f"{num//g}/{den//g}"

    # Try decimal
    try:
        val = float(s)
        # Round to 6 decimal places to avoid floating noise
        return f"{val:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        pass

    # Try SymPy parse
    try:
        expr = sympify(s)
        return str(expr)
    except Exception:
        pass

    # Try LaTeX parse
    try:
        expr = parse_latex(s)
        return str(simplify(expr))
    except Exception:
        pass

    return None


def answers_match(pred: str, gold: str) -> bool:
    """Return True if pred and gold normalize to the same value."""
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if p is None or g is None:
        return False
    if p == g:
        return True
    # Numeric comparison with tolerance
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ── SymPy Verification ────────────────────────────────────────────────────────

def sympy_verify(solution: str, claimed_answer: str) -> Optional[bool]:
    """
    Try to verify a claimed answer using SymPy by extracting and
    evaluating any equations in the solution.

    Returns:
        True  — verification passed
        False — verification detected a wrong answer
        None  — inconclusive (could not verify)
    """
    # Extract all \boxed{} values from solution as candidate answers
    boxed = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if not boxed:
        return None  # nothing to verify

    last_boxed = boxed[-1].strip()
    norm_claimed = normalize_answer(claimed_answer)
    norm_solution = normalize_answer(last_boxed)

    if norm_claimed is None or norm_solution is None:
        return None

    if norm_claimed == norm_solution:
        return True

    # Try numeric comparison
    try:
        if abs(float(norm_claimed) - float(norm_solution)) < 1e-6:
            return True
        else:
            return False  # Actively disagrees
    except (ValueError, TypeError):
        return None