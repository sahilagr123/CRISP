"""Extract \\boxed{} answers from math solutions."""
from __future__ import annotations

import re

# Strip <think>...</think> blocks so intermediate computations
# don't pollute fallback answer extraction.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Fallback patterns tried in order when \\boxed{} is absent.
_FALLBACK_PATTERNS = [
    # "the answer is 42", "final answer is 42", "answer: 42"
    re.compile(
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is|=|:)\s*"
        r"\$?(-?\d+(?:[./]\d+)?)\$?",
        re.IGNORECASE,
    ),
    # "= 42" at end of a line
    re.compile(r"=\s*\$?(-?\d+(?:[./]\d+)?)\$?\s*$", re.MULTILINE),
]


def extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in text.

    Handles nested braces by counting brace depth.
    Returns None if no \\boxed{} is found.
    """
    results = []
    search_str = "\\boxed{"
    idx = 0
    while idx < len(text):
        pos = text.find(search_str, idx)
        if pos == -1:
            break
        # Start after \boxed{
        start = pos + len(search_str)
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{" and (i == 0 or text[i - 1] != "\\"):
                depth += 1
            elif text[i] == "}" and (i == 0 or text[i - 1] != "\\"):
                depth -= 1
            i += 1
        if depth == 0:
            results.append(text[start : i - 1])
        idx = i
    return results[-1] if results else None


def _last_standalone_number(text: str) -> str | None:
    """Return the last number that appears alone on a line, within the final 3 lines."""
    lines = text.strip().splitlines()
    for line in reversed(lines[-3:]):
        line = line.strip().strip("$").strip()
        if re.fullmatch(r"-?\d+(?:[./]\d+)?", line):
            return line
    return None


def extract_answer(text: str, truncated: bool = False) -> str | None:
    """Extract answer from model output with fallback chain.

    1. Try \\boxed{} (strict, preferred).
    2. If not truncated, try "the answer is X" / "= X" patterns.
    3. If not truncated, try last standalone number on a line (final 3 lines).

    When truncated=True, only \\boxed{} is trusted — the model was cut off
    mid-reasoning so any trailing number is likely an intermediate result,
    not a final answer.

    Thinking-mode output (<think>...</think>) is stripped before fallback
    extraction so intermediate computations don't get picked as the answer.
    """
    # Strip thinking blocks for extraction — intermediate computations
    # inside <think> often contain "= 5184" etc. that confuse fallbacks.
    clean = _THINK_RE.sub("", text)

    result = extract_boxed(clean)
    if result is not None:
        return result

    # Don't trust fallbacks on truncated outputs — the model never
    # finished reasoning, so any number we find is unreliable.
    if truncated:
        return None

    for pattern in _FALLBACK_PATTERNS:
        matches = list(pattern.finditer(clean))
        if matches:
            return matches[-1].group(1).strip()

    return _last_standalone_number(clean)
