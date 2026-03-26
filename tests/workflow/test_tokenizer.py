"""Tests for tokenizer and prompt building."""
import sys
from unittest.mock import MagicMock, patch

from crisp.config import CRISPConfig


def _mock_tokenizer():
    tok = MagicMock()
    tok.encode.side_effect = lambda text, add_special_tokens=True: list(range(len(text.split())))
    tok.eos_token_id = 0
    return tok


def test_get_tokenizer_caches():
    """get_tokenizer returns the same instance on repeated calls."""
    from crisp.workflow.tokenizer import get_tokenizer, _tokenizer_cache

    _tokenizer_cache.clear()

    mock_transformers = MagicMock()
    mock_transformers.AutoTokenizer.from_pretrained.return_value = _mock_tokenizer()

    with patch.dict(sys.modules, {"transformers": mock_transformers}):
        t1 = get_tokenizer("model-a")
        t2 = get_tokenizer("model-a")
    assert t1 is t2
    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once()


def test_build_coach_prompts():
    """build_coach_prompts returns N token-ID lists from the template."""
    from crisp.workflow.tokenizer import build_coach_prompts

    tok = _mock_tokenizer()
    config = CRISPConfig()
    result = build_coach_prompts(tok, config, n=3)
    assert len(result) == 3
    assert all(isinstance(r, list) for r in result)
    assert all(isinstance(t, int) for t in result[0])


def test_coach_warmup_switches_system_prompt():
    """Three-phase warmup: AMC 10 → AMC 12 → pure accuracy-driven."""
    from crisp.workflow.tokenizer import build_coach_prompts

    captured_prompts = []

    def _capture_chat(msgs, **kw):
        captured_prompts.append(msgs[0]["content"])  # system prompt
        return [1, 2, 3]

    tok = MagicMock()
    tok.apply_chat_template.side_effect = _capture_chat
    config = CRISPConfig()
    config.coach.warmup_iters = 5
    config.coach.rampup_iters = 15

    # Phase 1 (iter < 5): AMC 10 anchor
    build_coach_prompts(tok, config, n=1, iteration=0)
    assert "AMC 10" in captured_prompts[-1]

    # Phase 2 (5 <= iter < 15): AMC 12 ramp-up
    build_coach_prompts(tok, config, n=1, iteration=5)
    assert "AMC 12" in captured_prompts[-1]

    # Phase 3 (iter >= 15): pure accuracy-driven
    build_coach_prompts(tok, config, n=1, iteration=15)
    assert "40-60%" in captured_prompts[-1]
    assert "AMC 10" not in captured_prompts[-1]
    assert "AMC 12" not in captured_prompts[-1]


def test_build_player_prompts():
    """build_player_prompts tokenizes each problem text."""
    from crisp.workflow.tokenizer import build_player_prompts
    from crisp.types import Problem

    tok = _mock_tokenizer()
    problems = [Problem(text="What is 2+2?", ground_truth="4")]
    config = CRISPConfig()
    result = build_player_prompts(tok, config, problems, player_id=0)
    assert len(result) == config.player.rollouts_per_problem
    assert all(isinstance(r, list) for r in result)


def test_build_discussion_prompts():
    """build_discussion_prompts tokenizes formatted discussion templates."""
    from crisp.workflow.tokenizer import build_discussion_prompts

    tok = _mock_tokenizer()
    tok.apply_chat_template.side_effect = lambda msgs, **kw: list(range(5))
    config = MagicMock()
    config.coach.discussion_system_prompt = "You are an evaluator."
    prompt_texts = ["Evaluate this...", "Another eval..."]
    result = build_discussion_prompts(tok, config, prompt_texts)
    assert len(result) == 2
    assert all(isinstance(r, list) for r in result)
    assert all(isinstance(t, int) for r in result for t in r)
