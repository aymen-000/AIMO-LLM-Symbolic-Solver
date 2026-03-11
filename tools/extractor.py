"""
Extracts executable code blocks and mathematical expressions
from a model's chain-of-thought response text.

Handles:
  - ```python ... ``` fenced code blocks
  - Inline SymPy expressions
  - LaTeX equations (converted to SymPy-parseable form)
  - Final answer extraction (delegated to answer_utils)
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class CodeBlock:
    language: str           # "python", "sympy", "math", "unknown"
    code: str               # raw code string
    line_start: int = -1    # position in original text


@dataclass
class MathExpression:
    raw: str                # original LaTeX or plain-math string
    sympy_form: str         # best-effort SymPy-parseable conversion
    context: str            # surrounding sentence for context


@dataclass
class ExtractionResult:
    code_blocks:       list[CodeBlock]      = field(default_factory=list)
    math_expressions:  list[MathExpression] = field(default_factory=list)
    claimed_answer:    Optional[str]        = None
    has_executable:    bool                 = False   # any runnable Python found


# ── LaTeX → SymPy conversion helpers ─────────────────────────────────────────

LATEX_TO_SYMPY = [
    # Fractions
    (r"\\frac\{([^}]+)\}\{([^}]+)\}",  r"(\1)/(\2)"),
    # Powers
    (r"\^\\{([^}]+)\\}",               r"**(\1)"),
    (r"\^(\w)",                         r"**\1"),
    (r"\^\{([^}]+)\}",                  r"**(\1)"),
    # Square root
    (r"\\sqrt\{([^}]+)\}",              r"sqrt(\1)"),
    (r"\\sqrt\s+(\w)",                  r"sqrt(\1)"),
    # Trig
    (r"\\sin\b",  "sin"),
    (r"\\cos\b",  "cos"),
    (r"\\tan\b",  "tan"),
    (r"\\log\b",  "log"),
    (r"\\ln\b",   "ln"),
    (r"\\pi\b",   "pi"),
    (r"\\infty\b","oo"),
    # Multiplication (implicit \cdot or \times)
    (r"\\cdot",   "*"),
    (r"\\times",  "*"),
    # Remove remaining LaTeX commands
    (r"\\[a-zA-Z]+\{([^}]*)\}", r"\1"),
    (r"\\[a-zA-Z]+",            ""),
    # Clean up
    (r"\s+",                    " "),
]

def latex_to_sympy(latex: str) -> str:
    """Best-effort LaTeX → SymPy string conversion."""
    s = latex.strip()
    for pattern, replacement in LATEX_TO_SYMPY:
        s = re.sub(pattern, replacement, s)
    return s.strip()


# ── Extractor ─────────────────────────────────────────────────────────────────

class CoTExtractor:
    """
    Extracts code blocks and math expressions from CoT response text.
    """

    # Fenced code block pattern: ```python ... ``` or ```sympy ... ```
    FENCED_CODE_RE = re.compile(
        r"```(\w*)\s*\n(.*?)```",
        re.DOTALL
    )

    # Inline code: `expr`
    INLINE_CODE_RE = re.compile(r"`([^`\n]{3,80})`")

    # Display math: $$ ... $$ or \[ ... \]
    DISPLAY_MATH_RE = re.compile(
        r"\$\$(.+?)\$\$|\\\[(.+?)\\\]",
        re.DOTALL
    )

    # Inline math: $ ... $
    INLINE_MATH_RE = re.compile(r"\$([^$\n]{2,60})\$")

    # Boxed answer
    BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

    # Explicit answer line
    ANSWER_LINE_RE = re.compile(r"Answer:\s*([^\n]+)", re.IGNORECASE)

    def extract(self, text: str) -> ExtractionResult:
        result = ExtractionResult()

        # ── Code blocks ───────────────────────────────────────────────────────
        for match in self.FENCED_CODE_RE.finditer(text):
            lang = match.group(1).lower() or "unknown"
            code = match.group(2).strip()
            if not code:
                continue
            block = CodeBlock(
                language=lang,
                code=code,
                line_start=text[:match.start()].count("\n"),
            )
            result.code_blocks.append(block)
            if lang in ("python", "sympy", "py", "unknown", ""):
                result.has_executable = True

        # Inline code that looks like Python (contains = or SymPy keywords)
        for match in self.INLINE_CODE_RE.finditer(text):
            code = match.group(1).strip()
            if any(kw in code for kw in ("solve", "simplify", "factor", "expand", "=", "sympy")):
                result.code_blocks.append(CodeBlock(
                    language="sympy_inline",
                    code=code,
                ))
                result.has_executable = True

        # ── Math expressions ──────────────────────────────────────────────────
        for match in self.DISPLAY_MATH_RE.finditer(text):
            latex = (match.group(1) or match.group(2) or "").strip()
            if latex:
                # Get surrounding context (50 chars each side)
                start = max(0, match.start() - 50)
                end   = min(len(text), match.end() + 50)
                result.math_expressions.append(MathExpression(
                    raw=latex,
                    sympy_form=latex_to_sympy(latex),
                    context=text[start:end].replace("\n", " "),
                ))

        for match in self.INLINE_MATH_RE.finditer(text):
            latex = match.group(1).strip()
            # Skip trivially short expressions like $n$, $x$
            if len(latex) >= 3:
                start = max(0, match.start() - 40)
                end   = min(len(text), match.end() + 40)
                result.math_expressions.append(MathExpression(
                    raw=latex,
                    sympy_form=latex_to_sympy(latex),
                    context=text[start:end].replace("\n", " "),
                ))

        # ── Claimed answer ────────────────────────────────────────────────────
        # Priority: explicit Answer: line > \boxed{} > last number
        m = self.ANSWER_LINE_RE.search(text)
        if m:
            result.claimed_answer = m.group(1).strip()
        else:
            m = self.BOXED_RE.search(text)
            if m:
                result.claimed_answer = m.group(1).strip()
            else:
                # Last standalone integer
                nums = re.findall(r"\b(\d{1,6})\b", text)
                if nums:
                    result.claimed_answer = nums[-1]

        return result


# ── Convenience function ──────────────────────────────────────────────────────

_extractor = CoTExtractor()

def extract_from_response(text: str) -> ExtractionResult:
    return _extractor.extract(text)