"""Parse post-discussion responses into EVALUATION and FINAL ANSWER segments."""
from __future__ import annotations

from typing import Optional, Tuple

from crisp.verifier.answer_extraction import extract_answer

EVALUATION_DELIMITER = "EVALUATION:"
FINAL_ANSWER_DELIMITER = "FINAL ANSWER:"


def parse_discussion_response(text: str) -> Tuple[str, Optional[str]]:
    """Parse a post-discussion response into evaluation text and final answer.

    Looks for the FINAL ANSWER: delimiter to split segments.
    If not found, entire text is treated as answer segment (empty evaluation).
    The final answer is extracted from \\boxed{} in the answer segment.

    Returns:
        (evaluation_text, extracted_answer) where extracted_answer may be None.
    """
    if not text:
        return "", None

    # Find the LAST occurrence of the delimiter
    idx = text.rfind(FINAL_ANSWER_DELIMITER)
    if idx == -1:
        # No delimiter: entire text is answer segment
        return "", extract_answer(text)

    evaluation_text = text[:idx]
    answer_segment = text[idx + len(FINAL_ANSWER_DELIMITER):]
    answer = extract_answer(answer_segment)

    # Strip the EVALUATION: prefix if present
    eval_stripped = evaluation_text.strip()
    if eval_stripped.startswith(EVALUATION_DELIMITER):
        evaluation_text = eval_stripped[len(EVALUATION_DELIMITER):]

    return evaluation_text, answer
