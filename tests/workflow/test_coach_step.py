"""Tests for coach_step — mock vLLM, real answer extraction."""
from unittest.mock import MagicMock, patch

import numpy as np

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem


def _make_ctx(**overrides):
    """Helper to create a WorkflowContext with mocked infra."""
    from crisp.workflow.context import WorkflowContext
    defaults = dict(
        player_vllm=[MagicMock()],
        coach_vllm=[MagicMock()],
        ref_model=MagicMock(),
        ds_alice=MagicMock(),
        ds_bob=MagicMock(),
        ds_coach=MagicMock(),
        config=CRISPConfig(),
        alice_ema=EMATracker(),
        bob_ema=EMATracker(),
        coach_ema=EMATracker(),
        rep_buffer=RepetitionBuffer(),
    )
    defaults.update(overrides)
    return WorkflowContext(**defaults)


def test_generate_problems_extracts_ground_truth():
    """generate_problems parses \\boxed{} from coach output as ground_truth."""
    from crisp.workflow.coach_step import generate_problems

    ctx = _make_ctx()

    from crisp.types import Rollout
    # Step 1: generate returns <question> tagged output (no answer)
    gen_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[1, 2, 3],
                text="<question>What is the value of 2 + 2?</question>",
                log_probs=[-0.1, -0.2, -0.3]),
        Rollout(problem_idx=1, player_id=-1, tokens=[4, 5, 6],
                text="<question>Solve the equation x^2 = 9 for positive x.</question>",
                log_probs=[-0.4, -0.5, -0.6]),
    ]
    # Step 2: resolve returns \boxed{} answers
    solve_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[7, 8],
                text="2+2=4. \\boxed{4}",
                log_probs=[-0.1, -0.2]),
        Rollout(problem_idx=1, player_id=-1, tokens=[9, 10],
                text="x=3. \\boxed{3}",
                log_probs=[-0.3, -0.4]),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(2, 384).astype(np.float32)

    with patch("crisp.workflow.coach_step.generate_samples",
               side_effect=[gen_rollouts, solve_rollouts]), \
         patch("crisp.workflow.coach_step._build_solve_prompts", return_value=[[1, 2], [3, 4]]), \
         patch("crisp.workflow.coach_step._get_embedder", return_value=mock_embedder):
        problems = generate_problems(ctx)

    assert len(problems) == 2
    assert isinstance(problems[0], Problem)
    assert problems[0].ground_truth == "4"
    assert problems[1].ground_truth == "3"
    assert problems[0].text == "What is the value of 2 + 2?"
    assert problems[0].coach_embedding is not None
    assert problems[0].coach_embedding.shape == (384,)


def test_generate_problems_fallback_answer_extraction():
    """generate_problems uses extract_answer fallbacks (not just \\boxed{})."""
    from crisp.workflow.coach_step import generate_problems
    from crisp.types import Rollout

    ctx = _make_ctx()

    # Step 1: both questions parse fine
    gen_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[1, 2],
                text="<question>What is the value of 2 + 2?</question>",
                log_probs=[-0.1, -0.2]),
        Rollout(problem_idx=1, player_id=-1, tokens=[3, 4],
                text="<question>Find the value of x if 3x = 21.</question>",
                log_probs=[-0.3, -0.4]),
        Rollout(problem_idx=2, player_id=-1, tokens=[5, 6],
                text="<question>What is the result of 10 divided by 2?</question>",
                log_probs=[-0.5, -0.6]),
    ]
    # Step 2: various answer formats
    solve_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[7, 8],
                text="The answer is 4",  # fallback: "the answer is X"
                log_probs=[-0.1, -0.2]),
        Rollout(problem_idx=1, player_id=-1, tokens=[9, 10],
                text="x=7. \\boxed{7}",  # strict: \boxed{}
                log_probs=[-0.3, -0.4]),
        Rollout(problem_idx=2, player_id=-1, tokens=[11, 12],
                text="I'm not sure about this one, let me think more...",  # no answer at all
                log_probs=[-0.5, -0.6]),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(3, 384).astype(np.float32)

    with patch("crisp.workflow.coach_step.generate_samples",
               side_effect=[gen_rollouts, solve_rollouts]), \
         patch("crisp.workflow.coach_step._build_solve_prompts", return_value=[[1, 2], [3, 4], [5, 6]]), \
         patch("crisp.workflow.coach_step._get_embedder", return_value=mock_embedder):
        problems = generate_problems(ctx)

    # All 3 problems returned
    assert len(problems) == 3
    # "The answer is 4" — parsed by fallback
    assert problems[0].self_solvable is True
    assert problems[0].ground_truth == "4"
    # \boxed{7} — parsed by strict extraction
    assert problems[1].self_solvable is True
    assert problems[1].ground_truth == "7"
    # No answer at all — marked unsolvable
    assert problems[2].self_solvable is False
    assert problems[2].ground_truth == "UNSOLVABLE"


def test_generate_problems_coach_sequence_preserved():
    """generate_problems stores the generation (step 1) token sequence on Problem."""
    from crisp.workflow.coach_step import generate_problems
    from crisp.types import Rollout

    ctx = _make_ctx()

    # Step 1: generation sequence (this is what gets stored for training)
    gen_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[10, 20, 30],
                text="<question>What is the product of 6 and 7?</question>",
                log_probs=[-0.1, -0.2, -0.3]),
    ]
    # Step 2: resolve sequence (used only for ground truth extraction)
    solve_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[40, 50],
                text="6*7=42. \\boxed{42}",
                log_probs=[-0.4, -0.5]),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

    with patch("crisp.workflow.coach_step.generate_samples",
               side_effect=[gen_rollouts, solve_rollouts]), \
         patch("crisp.workflow.coach_step._build_solve_prompts", return_value=[[1, 2]]), \
         patch("crisp.workflow.coach_step._get_embedder", return_value=mock_embedder):
        problems = generate_problems(ctx)

    assert problems[0].coach_sequence is not None
    # Should preserve the generation (step 1) tokens, not solve tokens
    assert problems[0].coach_sequence.tokens == [10, 20, 30]
    assert problems[0].coach_sequence.log_probs == [-0.1, -0.2, -0.3]


def test_extract_question_tagless_fallback():
    """_extract_question falls back to raw text when <question> tags are missing."""
    from crisp.workflow.coach_step import _extract_question

    # Normal tagged output
    assert _extract_question("<question>What is 2+2? Solve step by step.</question>") == \
        "What is 2+2? Solve step by step."

    # Tagless: raw math problem (coach format drift)
    result = _extract_question("Find the value of x if 3x + 7 = 22. Express your answer as an integer.")
    assert result is not None
    assert "3x + 7 = 22" in result

    # Tagless with preamble
    result = _extract_question("Here's a math problem: Find the sum of all integers from 1 to 100.")
    assert result is not None
    assert "sum" in result.lower()

    # Too short — rejected even in fallback
    assert _extract_question("short text") is None

    # Template placeholder — rejected
    assert _extract_question("[PROBLEM]") is None


def test_generate_problems_tagless_fallback():
    """generate_problems accepts coach output without <question> tags."""
    from crisp.workflow.coach_step import generate_problems
    from crisp.types import Rollout

    ctx = _make_ctx()

    # Coach outputs valid problems but without tags (format drift)
    gen_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[1, 2, 3],
                text="Find the value of x if 5x - 3 = 12. Express your answer as an integer.",
                log_probs=[-0.1, -0.2, -0.3]),
    ]
    solve_rollouts = [
        Rollout(problem_idx=0, player_id=-1, tokens=[7, 8],
                text="5x=15, x=3. \\boxed{3}",
                log_probs=[-0.1, -0.2]),
    ]

    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.randn(1, 384).astype(np.float32)

    with patch("crisp.workflow.coach_step.generate_samples",
               side_effect=[gen_rollouts, solve_rollouts]), \
         patch("crisp.workflow.coach_step._build_solve_prompts", return_value=[[1, 2]]), \
         patch("crisp.workflow.coach_step._get_embedder", return_value=mock_embedder):
        problems = generate_problems(ctx)

    assert len(problems) == 1
    assert problems[0].ground_truth == "3"
    assert "5x" in problems[0].text


def test_build_coach_prompts_uses_tokenizer():
    """_build_coach_prompts uses tokenizer to build prompts."""
    from crisp.workflow.coach_step import _build_coach_prompts

    ctx = _make_ctx()
    ctx.tokenizer = MagicMock()
    ctx.tokenizer.encode.side_effect = lambda text, add_special_tokens=True: [1, 2, 3]
    ctx.config.coach.batch_size = 2
    ctx.config.coach.coach_prompt_template = "Generate a {topic} math problem"

    result = _build_coach_prompts(ctx)
    assert len(result) == 2
    assert all(isinstance(r, list) for r in result)
    assert all(isinstance(t, int) for t in result[0])
