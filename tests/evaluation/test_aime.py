"""Tests for AIME 2024/2025 dataset loaders."""
from unittest.mock import patch, MagicMock

from crisp.types import Problem


def _make_mock_aime24():
    """Mock math-ai/aime24 dataset."""
    rows = [
        {"id": 60, "problem": "Find x if 2x=10.", "solution": "\\boxed{5}",
         "url": "https://example.com/1"},
        {"id": 61, "problem": "Find m+n if m/n=3/7.", "solution": "\\boxed{10}",
         "url": "https://example.com/2"},
    ]
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(rows)
    mock_ds.__len__ = lambda self: len(rows)
    return mock_ds


def _make_mock_aime25():
    """Mock math-ai/aime25 dataset."""
    rows = [
        {"id": "0", "problem": "Find the sum of bases.", "answer": "70"},
        {"id": "1", "problem": "Find the area.", "answer": "588"},
    ]
    mock_ds = MagicMock()
    mock_ds.__iter__ = lambda self: iter(rows)
    mock_ds.__len__ = lambda self: len(rows)
    return mock_ds


def test_load_aime24_problems():
    """load_aime24_problems returns Problems with boxed answers extracted."""
    import crisp.evaluation.aime as mod
    mod._aime24_cache = None

    with patch("crisp.evaluation.aime.load_dataset", return_value=_make_mock_aime24()):
        problems = mod.load_aime24_problems()

    assert len(problems) == 2
    assert all(isinstance(p, Problem) for p in problems)
    assert problems[0].text == "Find x if 2x=10."
    assert problems[0].ground_truth == "5"  # Extracted from \boxed{5}
    assert problems[1].ground_truth == "10"
    mod._aime24_cache = None


def test_load_aime25_problems():
    """load_aime25_problems returns Problems with plain string answers."""
    import crisp.evaluation.aime as mod
    mod._aime25_cache = None

    with patch("crisp.evaluation.aime.load_dataset", return_value=_make_mock_aime25()):
        problems = mod.load_aime25_problems()

    assert len(problems) == 2
    assert all(isinstance(p, Problem) for p in problems)
    assert problems[0].text == "Find the sum of bases."
    assert problems[0].ground_truth == "70"
    assert problems[1].ground_truth == "588"
    mod._aime25_cache = None


def test_aime24_caches():
    """Repeated calls return cached list."""
    import crisp.evaluation.aime as mod
    mod._aime24_cache = None

    with patch("crisp.evaluation.aime.load_dataset", return_value=_make_mock_aime24()) as mock_load:
        p1 = mod.load_aime24_problems()
        p2 = mod.load_aime24_problems()

    mock_load.assert_called_once()
    assert p1 is p2
    mod._aime24_cache = None


def test_extract_boxed():
    """_extract_boxed handles various formats."""
    from crisp.evaluation.aime import _extract_boxed

    assert _extract_boxed("\\boxed{42}") == "42"
    assert _extract_boxed("\\boxed{3/7}") == "3/7"
    assert _extract_boxed("plain 42") == "plain 42"
    assert _extract_boxed("  123  ") == "123"
