"""Tests for rollout_step — mock vLLM, real domain logic."""
from unittest.mock import MagicMock, patch

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import Problem, Rollout


def _make_ctx(**overrides):
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


def test_generate_rollouts_correct_answer():
    """Rollouts with correct \\boxed{} answers get reward=1.0."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="What is 2+2?", ground_truth="4")]

    mock_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="The answer is \\boxed{4}",
                log_probs=[-0.1, -0.2]),
        Rollout(problem_idx=0, player_id=0, tokens=[3, 4], text="I think \\boxed{5}",
                log_probs=[-0.3, -0.4]),
    ]

    with patch("crisp.workflow.rollout_step.generate_samples", return_value=mock_rollouts):
        rollouts = generate_rollouts(ctx, problems, player_id=0)

    assert rollouts[0].answer == "4"
    assert rollouts[0].correct is True
    assert rollouts[0].reward == 1.0

    assert rollouts[1].answer == "5"
    assert rollouts[1].correct is False
    assert rollouts[1].reward == 0.0


def test_generate_rollouts_no_boxed_fallback():
    """Rollouts without \\boxed{} but with 'the answer is X' get extracted via fallback."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="What is 1+1?", ground_truth="2")]

    mock_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="The answer is 2",
                log_probs=[-0.1, -0.2]),
    ]

    with patch("crisp.workflow.rollout_step.generate_samples", return_value=mock_rollouts):
        rollouts = generate_rollouts(ctx, problems, player_id=0)

    assert rollouts[0].answer == "2"
    assert rollouts[0].correct is True
    assert rollouts[0].reward == 1.0


def test_generate_rollouts_truly_unparseable():
    """Rollouts with no extractable answer get answer=None, reward=-0.5."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="What is 1+1?", ground_truth="2")]

    mock_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2],
                text="I'm not sure how to solve this problem",
                log_probs=[-0.1, -0.2]),
    ]

    with patch("crisp.workflow.rollout_step.generate_samples", return_value=mock_rollouts):
        rollouts = generate_rollouts(ctx, problems, player_id=0)

    assert rollouts[0].answer is None
    assert rollouts[0].correct is None
    assert rollouts[0].reward == -0.5


def test_generate_rollouts_overlong_penalty():
    """Overlong rollouts get penalty subtracted from reward."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="Q", ground_truth="1")]

    long_tokens = list(range(9000))  # > l_max=8192
    mock_rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=long_tokens,
                text="\\boxed{1}", log_probs=[0.0] * 9000),
    ]

    with patch("crisp.workflow.rollout_step.generate_samples", return_value=mock_rollouts):
        rollouts = generate_rollouts(ctx, problems, player_id=0)

    assert rollouts[0].correct is True
    # reward = 1.0 (correct) - overlong_penalty(9000, 8192, 2048) ≈ 0.605
    assert rollouts[0].reward < 1.0
    assert rollouts[0].reward > 0.0


def test_generate_all_rollouts_calls_both_players():
    """generate_all_rollouts returns dict keyed by player ID for both players."""
    from crisp.workflow.rollout_step import generate_all_rollouts

    ctx = _make_ctx()
    problems = [Problem(text="What is 2+2?", ground_truth="4")]

    called_player_ids = []

    def fake_generate_samples(engine, prompt_token_ids, problem_indices, player_id, **kwargs):
        called_player_ids.append(player_id)
        return [
            Rollout(problem_idx=0, player_id=player_id, tokens=[1, 2],
                    text="\\boxed{4}", log_probs=[-0.1, -0.2]),
        ]

    with patch("crisp.workflow.rollout_step.generate_samples", side_effect=fake_generate_samples):
        result = generate_all_rollouts(ctx, problems)

    assert set(called_player_ids) == {0, 1}
    assert set(result.keys()) == {0, 1}
    assert result[0][0].player_id == 0
    assert result[1][0].player_id == 1


def test_problem_indices_grouped_by_rpg():
    """problem_indices must match build_player_prompts grouping: rpg copies per problem."""
    from crisp.workflow.rollout_step import generate_rollouts

    ctx = _make_ctx()
    ctx.config.player.rollouts_per_problem = 3
    problems = [
        Problem(text="P0", ground_truth="10"),
        Problem(text="P1", ground_truth="20"),
    ]

    captured_indices = {}

    def fake_gen(engine, prompt_token_ids, problem_indices, player_id, **kwargs):
        captured_indices["pi"] = list(problem_indices)
        return [
            Rollout(problem_idx=pi, player_id=player_id, tokens=[1],
                    text="\\boxed{10}", log_probs=[0.0])
            for pi in problem_indices
        ]

    # 6 prompts: [P0, P0, P0, P1, P1, P1]
    prompts = [[1, 2]] * 6
    with patch("crisp.workflow.rollout_step.generate_samples", side_effect=fake_gen):
        generate_rollouts(ctx, problems, player_id=0, prompt_token_ids=prompts)

    assert captured_indices["pi"] == [0, 0, 0, 1, 1, 1]


def test_generate_all_rollouts_builds_prompts_from_tokenizer():
    """generate_all_rollouts uses tokenizer when no prompts given."""
    from crisp.workflow.rollout_step import generate_all_rollouts

    ctx = _make_ctx()
    ctx.tokenizer = MagicMock()
    ctx.tokenizer.encode.side_effect = lambda text, add_special_tokens=True: [1, 2]
    ctx.config.player.rollouts_per_problem = 2

    problems = [Problem(text="What is 2+2?", ground_truth="4")]

    with patch("crisp.workflow.rollout_step.generate_rollouts") as mock_gen:
        mock_gen.return_value = []
        generate_all_rollouts(ctx, problems)

    # Should have been called with prompt_token_ids (not None)
    for call in mock_gen.call_args_list:
        assert call.kwargs.get('prompt_token_ids') is not None or call[1].get('prompt_token_ids') is not None
