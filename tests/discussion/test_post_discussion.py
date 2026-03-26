"""Tests for EVALUATION / FINAL ANSWER parsing."""
import pytest

from crisp.discussion.post_discussion import parse_discussion_response


class TestParseDiscussionResponse:
    def test_both_segments(self):
        text = (
            "EVALUATION: Alice's solution has an error in step 3.\n"
            "FINAL ANSWER: \\boxed{42}"
        )
        eval_text, answer = parse_discussion_response(text)
        assert "error in step 3" in eval_text
        assert answer == "42"

    def test_no_delimiter_entire_text_is_answer(self):
        text = "The answer is \\boxed{42}"
        eval_text, answer = parse_discussion_response(text)
        assert eval_text == ""
        assert answer == "42"

    def test_empty_evaluation(self):
        text = "EVALUATION:\nFINAL ANSWER: \\boxed{7}"
        eval_text, answer = parse_discussion_response(text)
        assert eval_text.strip() == ""
        assert answer == "7"

    def test_no_boxed_in_final_answer(self):
        """Fallback extraction recovers '42' even without \\boxed{}."""
        text = "EVALUATION: good\nFINAL ANSWER: 42"
        eval_text, answer = parse_discussion_response(text)
        assert "good" in eval_text
        assert answer == "42"

    def test_multiple_final_answer_delimiters(self):
        text = (
            "EVALUATION: I think FINAL ANSWER: is mentioned here\n"
            "FINAL ANSWER: \\boxed{5}"
        )
        eval_text, answer = parse_discussion_response(text)
        # Should split on the LAST occurrence
        assert answer == "5"

    def test_empty_response(self):
        eval_text, answer = parse_discussion_response("")
        assert eval_text == ""
        assert answer is None
