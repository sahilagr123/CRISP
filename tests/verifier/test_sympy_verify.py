"""Tests for 3-strategy SymPy verification."""
import pytest

from crisp.verifier.sympy_verify import check, equivalent


class TestCheck:
    """Test check(answer, ground_truth) -> bool."""

    # Strategy 1: exact string match
    def test_exact_match(self):
        assert check("42", "42") is True

    def test_exact_mismatch(self):
        assert check("43", "42") is False

    # Strategy 2: numeric comparison
    def test_numeric_float_equivalence(self):
        assert check("0.333333", "1/3") is True

    def test_numeric_integer_vs_float(self):
        assert check("3.0", "3") is True

    def test_numeric_tolerance(self):
        assert check("3.14159", "3.14159265") is True

    def test_numeric_outside_tolerance(self):
        assert check("3.15", "3.14") is False

    # Strategy 3: symbolic equivalence
    def test_symbolic_expand(self):
        assert check("x^2 + 2*x + 1", "(x+1)^2") is True

    def test_symbolic_fraction_simplify(self):
        assert check("\\frac{2}{4}", "\\frac{1}{2}") is True

    def test_symbolic_sqrt(self):
        assert check("\\sqrt{4}", "2") is True

    # Edge cases
    def test_none_answer(self):
        assert check(None, "42") is False

    def test_empty_answer(self):
        assert check("", "42") is False

    def test_both_none(self):
        assert check(None, None) is False

    def test_whitespace_handling(self):
        assert check(" 42 ", "42") is True


class TestEquivalent:
    """Test equivalent(a, b) -> bool. Symmetric version."""

    def test_symmetric(self):
        assert equivalent("1/3", "0.333333") is True
        assert equivalent("0.333333", "1/3") is True

    def test_both_wrong_but_equal(self):
        assert equivalent("99", "99") is True

    def test_different_answers(self):
        assert equivalent("42", "43") is False

    def test_none_inputs(self):
        assert equivalent(None, "42") is False
        assert equivalent("42", None) is False
        assert equivalent(None, None) is False

    def test_latex_fraction_vs_decimal(self):
        """\\frac{1}{2} should be equivalent to 0.5."""
        assert equivalent("\\frac{1}{2}", "0.5") is True
        assert equivalent("0.5", "\\frac{1}{2}") is True

    def test_dfrac_equivalent_to_frac(self):
        """\\dfrac{a}{b} (display-style) should match \\frac{a}{b}."""
        assert equivalent("\\dfrac{7}{2}", "\\frac{7}{2}") is True
        assert equivalent("\\dfrac{7}{2}", "3.5") is True

    def test_numeric_symmetry_large_values(self):
        """equivalent(a, b) must equal equivalent(b, a) for large numbers.

        Regression: _numeric_equal used tol * max(1.0, abs(val_b)) which
        made the tolerance depend on argument order.
        """
        # At the boundary: abs(10001-10000)=1, tol*10000=1.0 (exact boundary)
        # Old code: check(10001,10000)=False, check(10000,10001)=True
        assert equivalent("10001", "10000") == equivalent("10000", "10001")
