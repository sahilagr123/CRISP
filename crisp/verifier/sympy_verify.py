"""Three-strategy math answer verification using SymPy."""
from __future__ import annotations

import re
from typing import Optional

import sympy
from sympy.parsing.latex import parse_latex


def check(answer: Optional[str], ground_truth: Optional[str]) -> bool:
    """Check if answer matches ground_truth using 3 strategies.

    Strategies tried in order:
    1. Exact string match (after stripping whitespace)
    2. Numeric comparison (within relative tolerance 1e-4)
    3. SymPy symbolic equivalence
    """
    if answer is None or ground_truth is None:
        return False
    answer = answer.strip()
    ground_truth = ground_truth.strip()
    if not answer or not ground_truth:
        return False

    # Strategy 1: exact string match
    if answer == ground_truth:
        return True

    # Strategy 2: numeric comparison
    if _numeric_equal(answer, ground_truth):
        return True

    # Strategy 3: symbolic equivalence
    if _symbolic_equal(answer, ground_truth):
        return True

    return False


def equivalent(a: Optional[str], b: Optional[str]) -> bool:
    """Symmetric equivalence check between two answers.

    Same logic as check() but semantically distinct:
    neither argument is privileged as 'ground truth'.
    """
    return check(a, b)


def _numeric_equal(a: str, b: str, tol: float = 1e-4) -> bool:
    """Try to parse both as numbers and compare within tolerance."""
    val_a = _try_parse_number(a)
    val_b = _try_parse_number(b)
    if val_a is not None and val_b is not None:
        return abs(val_a - val_b) < tol * max(1.0, abs(val_a), abs(val_b))
    return False


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a float, handling fractions."""
    s = s.strip()
    # Direct float
    try:
        return float(s)
    except ValueError:
        pass
    # Simple fraction: a/b
    m = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    # LaTeX fraction: \frac{a}{b} or \dfrac{a}{b}
    m = re.match(r"^\\d?frac\{(-?\d+)\}\{(-?\d+)\}$", s)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        if den != 0:
            return num / den
    return None


def _symbolic_equal(a: str, b: str) -> bool:
    """Try SymPy symbolic equivalence."""
    try:
        expr_a = _parse_to_sympy(a)
        expr_b = _parse_to_sympy(b)
        if expr_a is None or expr_b is None:
            return False
        diff = sympy.simplify(expr_a - expr_b)
        return diff == 0
    except (sympy.SympifyError, TypeError, ValueError, AttributeError):
        return False


def _parse_to_sympy(s: str) -> Optional[sympy.Expr]:
    """Parse a string (possibly LaTeX) to a SymPy expression."""
    s = s.strip()
    # Pre-process common LaTeX constructs to SymPy-friendly form
    s_py = _latex_to_sympy_str(s)
    # Try direct sympify first
    try:
        return sympy.sympify(s_py)
    except (sympy.SympifyError, TypeError, ValueError):
        pass
    # Try LaTeX parsing (requires antlr4)
    try:
        return parse_latex(s)
    except Exception:
        pass
    return None


def _latex_to_sympy_str(s: str) -> str:
    """Convert common LaTeX notation to SymPy-parseable strings."""
    s = s.replace("^", "**")
    # \sqrt{...} → sqrt(...)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # \frac{a}{b} or \dfrac{a}{b} → (a)/(b)
    s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    # \pi → pi, \infty → oo
    s = s.replace("\\pi", "pi").replace("\\infty", "oo")
    return s
