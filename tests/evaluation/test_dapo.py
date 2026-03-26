"""Tests for DAPO-17k dataset loader."""
from unittest.mock import patch, MagicMock

from crisp.types import Problem


def _make_mock_dataset():
    """Create a mock HuggingFace dataset with DAPO-17k structure."""
    rows = [
        {"prompt": "What is 2+2?", "solution": "4"},
        {"prompt": "What is 3*5?", "solution": "15"},
        {"prompt": "What is 10-7?", "solution": "3"},
    ]
    mock_ds = MagicMock()
    mock_ds.__getitem__ = lambda self, key: rows[key] if isinstance(key, int) else [r[key] for r in rows]
    mock_ds.__len__ = lambda self: len(rows)
    mock_ds.__iter__ = lambda self: iter(rows)
    return {"train": mock_ds}


def test_load_dapo_problems_returns_problems():
    """load_dapo_problems returns a list of Problem objects."""
    from crisp.evaluation.dapo import load_dapo_problems

    with patch("crisp.evaluation.dapo.load_dataset", return_value=_make_mock_dataset()):
        problems = load_dapo_problems()

    assert len(problems) == 3
    assert all(isinstance(p, Problem) for p in problems)
    assert problems[0].text == "What is 2+2?"
    assert problems[0].ground_truth == "4"


def test_load_dapo_problems_max_problems():
    """load_dapo_problems respects max_problems limit."""
    from crisp.evaluation.dapo import load_dapo_problems

    with patch("crisp.evaluation.dapo.load_dataset", return_value=_make_mock_dataset()):
        problems = load_dapo_problems(max_problems=2)

    assert len(problems) == 2


def test_load_dapo_problems_caches():
    """Repeated calls return the same cached list."""
    from crisp.evaluation.dapo import load_dapo_problems

    # Clear cache first
    import crisp.evaluation.dapo as mod
    mod._dapo_cache = None

    with patch("crisp.evaluation.dapo.load_dataset", return_value=_make_mock_dataset()) as mock_load:
        p1 = load_dapo_problems()
        p2 = load_dapo_problems()

    mock_load.assert_called_once()
    assert p1 is p2
    # Clean up
    mod._dapo_cache = None
