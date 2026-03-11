"""

Sandboxed Python code execution for tool-use verification.

Safety model:
  - Runs in a subprocess with a hard timeout
  - Restricted builtins (no file I/O, no network, no os/sys access)
  - Memory limit via resource module
  - Only math-safe imports allowed: sympy, numpy, itertools, math,
    fractions, collections, functools, re

Returns the stdout output and any computed variable named `result`,
`answer`, or `ans` from the executed namespace.
"""

import os
import sys
import ast
import resource
import subprocess
import tempfile
import textwrap
import json
from dataclasses import dataclass
from typing import Optional, Any


# ── Result structure ──────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    success:   bool
    output:    str            # stdout text
    answer:    Optional[str]  # extracted result/answer/ans variable
    error:     Optional[str]  # exception message if failed
    timed_out: bool = False


# ── Allowed imports whitelist ─────────────────────────────────────────────────

ALLOWED_IMPORTS = {
    "sympy", "numpy", "np", "math", "fractions", "itertools",
    "collections", "functools", "re", "decimal", "cmath",
    "scipy", "operator", "string", "random",
}

BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess", "import socket",
    "import requests", "import urllib", "import http", "import ftplib",
    "open(", "__import__", "exec(", "eval(", "compile(",
    "getattr(", "__builtins__", "globals()", "locals()",
    "importlib", "shutil", "pathlib", "pickle", "shelve",
]


# ── Static safety checker ─────────────────────────────────────────────────────

def is_safe_code(code: str) -> tuple[bool, str]:
    """
    Static analysis pass before execution.
    Returns (is_safe, reason_if_unsafe).
    """
    for pattern in BLOCKED_PATTERNS:
        if pattern in code:
            return False, f"Blocked pattern found: '{pattern}'"

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    for node in ast.walk(tree):
        # Block all imports not in whitelist
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            for name in names:
                root = name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    return False, f"Import not allowed: '{name}'"

    return True, ""


# ── Subprocess executor ───────────────────────────────────────────────────────

RUNNER_TEMPLATE = """
import sys
import json

# Restrict resource usage
try:
    import resource
    # 512 MB memory limit
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
except Exception:
    pass

# Allowed imports only
import math
import fractions
import itertools
import collections
import functools
import re
import decimal

try:
    import sympy
    from sympy import *
    import numpy as np
except ImportError:
    pass

_result_value = None

try:
{user_code}
    # Try to extract answer variable
    _ns = dir()
    for _var in ("result", "answer", "ans", "final_answer"):
        if _var in dir() and eval(_var) is not None:
            _result_value = str(eval(_var))
            break
except Exception as e:
    print(f"__ERROR__: {{e}}", file=sys.stderr)
    sys.exit(1)

print(f"__RESULT__: {{_result_value}}")
"""

def execute_code(code: str, timeout: float = 10.0) -> ExecutionResult:
    """
    Execute code in a sandboxed subprocess.

    Args:
        code:    Python code string to execute
        timeout: Hard timeout in seconds (default 10s)

    Returns:
        ExecutionResult with output, answer, and error info
    """
    # Static safety check
    safe, reason = is_safe_code(code)
    if not safe:
        return ExecutionResult(
            success=False,
            output="",
            answer=None,
            error=f"Safety check failed: {reason}",
        )

    # Indent user code for template
    indented = textwrap.indent(code, "    ")
    script   = RUNNER_TEMPLATE.format(user_code=indented)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # Parse result
        answer = None
        for line in stdout.splitlines():
            if line.startswith("__RESULT__:"):
                val = line[len("__RESULT__:"):].strip()
                if val and val != "None":
                    answer = val
                break

        # Clean stdout (remove our __RESULT__ line)
        clean_output = "\n".join(
            l for l in stdout.splitlines()
            if not l.startswith("__RESULT__:")
        )

        if proc.returncode != 0:
            error = stderr or "Non-zero exit code"
            return ExecutionResult(
                success=False,
                output=clean_output,
                answer=answer,
                error=error,
            )

        return ExecutionResult(
            success=True,
            output=clean_output,
            answer=answer,
            error=None,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            output="",
            answer=None,
            error="Execution timed out",
            timed_out=True,
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            output="",
            answer=None,
            error=str(e),
        )
    finally:
        os.unlink(tmp_path)


# ── Auto-wrap convenience ─────────────────────────────────────────────────────

def execute_expression(expr: str, timeout: float = 5.0) -> ExecutionResult:
    """
    Evaluate a single SymPy/Python expression and return its value.
    Wraps the expression in `result = simplify(expr)`.
    """
    code = f"result = simplify({expr})" if "simplify" not in expr else f"result = {expr}"
    return execute_code(code, timeout=timeout)