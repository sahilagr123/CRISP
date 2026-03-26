"""Tests for \\boxed{} answer extraction and fallback chain."""
import pytest

from crisp.verifier.answer_extraction import extract_answer, extract_boxed


class TestExtractBoxed:
    def test_simple_number(self):
        assert extract_boxed("The answer is \\boxed{42}") == "42"

    def test_fraction(self):
        assert extract_boxed("Therefore \\boxed{\\frac{3}{4}}") == "\\frac{3}{4}"

    def test_nested_braces(self):
        assert extract_boxed("\\boxed{\\{1, 2, 3\\}}") == "\\{1, 2, 3\\}"

    def test_latex_commands_inside(self):
        assert extract_boxed("\\boxed{\\sqrt{2} + \\pi}") == "\\sqrt{2} + \\pi"

    def test_no_boxed_returns_none(self):
        assert extract_boxed("There is no answer here") is None

    def test_empty_boxed(self):
        assert extract_boxed("\\boxed{}") == ""

    def test_multiple_boxed_returns_last(self):
        text = "First \\boxed{1} then \\boxed{2}"
        assert extract_boxed(text) == "2"

    def test_deeply_nested_braces(self):
        assert extract_boxed("\\boxed{f(g(x))}") == "f(g(x))"

    def test_multiline_text(self):
        text = "Step 1: compute\nStep 2: simplify\n\\boxed{7}"
        assert extract_boxed(text) == "7"

    def test_boxed_with_spaces(self):
        assert extract_boxed("\\boxed{ 42 }") == " 42 "

    def test_negative_number(self):
        assert extract_boxed("\\boxed{-3}") == "-3"

    def test_expression_with_equals(self):
        assert extract_boxed("\\boxed{x = 5}") == "x = 5"

    def test_unclosed_boxed_returns_none(self):
        """\\boxed{ with no closing brace should return None."""
        assert extract_boxed("\\boxed{42") is None

    def test_only_backslash_boxed_no_brace(self):
        """\\boxed without opening brace returns None."""
        assert extract_boxed("\\boxed 42") is None


class TestExtractAnswer:
    """Tests for the fallback answer extraction chain."""

    def test_boxed_preferred(self):
        """\\boxed{} is always preferred over fallback patterns."""
        assert extract_answer("The answer is 99. \\boxed{42}") == "42"

    def test_fallback_the_answer_is(self):
        assert extract_answer("After simplifying, the answer is 42.") == "42"

    def test_fallback_final_answer_is(self):
        assert extract_answer("So the final answer is 100") == "100"

    def test_fallback_answer_colon(self):
        assert extract_answer("Step 1: blah\nStep 2: blah\nAnswer: 7") == "7"

    def test_fallback_equals_end_of_line(self):
        assert extract_answer("x + 3 = 5\nSo x = 2") == "2"

    def test_fallback_standalone_number_last_line(self):
        assert extract_answer("Long reasoning here...\nTherefore,\n42") == "42"

    def test_fallback_negative_number(self):
        assert extract_answer("The answer is -3") == "-3"

    def test_fallback_fraction(self):
        assert extract_answer("The answer is 3/4") == "3/4"

    def test_fallback_dollar_signs(self):
        assert extract_answer("The answer is $42$") == "42"

    def test_no_answer_at_all(self):
        assert extract_answer("I don't know how to solve this") is None

    def test_standalone_number_not_too_greedy(self):
        """Only checks last 3 lines for standalone numbers."""
        text = "42\nLots of reasoning\nMore reasoning\nEven more\nNo answer"
        assert extract_answer(text) is None

    def test_last_match_wins_for_answer_is(self):
        """When 'the answer is X' appears twice, use the last one."""
        text = "The answer is 5.\nWait, I made an error.\nThe answer is 10."
        assert extract_answer(text) == "10"

    def test_thinking_block_stripped(self):
        """Intermediate computations in <think> blocks should be ignored."""
        text = (
            "<think>\nLet me compute 72^2 = 5184.\n"
            "Actually the radius is 3.\n</think>\n"
            "The answer is 3."
        )
        assert extract_answer(text) == "3"

    def test_thinking_block_boxed_outside(self):
        """\\boxed{} outside <think> is preferred over numbers inside."""
        text = (
            "<think>\nSo x = 5184 and y = 42.\n</think>\n"
            "\\boxed{3}"
        )
        assert extract_answer(text) == "3"

    def test_thinking_block_no_answer_outside(self):
        """If answer is only inside <think>, it should not be found."""
        text = "<think>\nThe answer is 5184.\n</think>\nI cannot determine the answer."
        assert extract_answer(text) is None
