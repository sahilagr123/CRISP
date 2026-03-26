"""Tests for discussion_step — mock vLLM, real discussion logic."""
from unittest.mock import MagicMock, patch

from crisp.config import CRISPConfig
from crisp.rewards.ema_tracker import EMATracker
from crisp.rewards.repetition_buffer import RepetitionBuffer
from crisp.types import DiscussionResult, Problem, Rollout


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


def test_run_discussion_no_disagreement():
    """No discussion when both players agree."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [Problem(text="2+2?", ground_truth="4")]

    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{4}",
                     log_probs=[-0.1], answer="4", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[2], text="\\boxed{4}",
                     log_probs=[-0.2], answer="4", correct=True, reward=1.0)],
    }

    disc_results, majority_answers = run_discussion(ctx, rollouts, problems)
    assert len(disc_results) == 0 or all(len(v) == 0 for v in disc_results.values())
    assert majority_answers[(0, 0)] == "4"
    assert majority_answers[(1, 0)] == "4"


def test_run_discussion_disagreement_triggers():
    """Discussion triggered when players disagree, results parsed correctly."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [Problem(text="What is 5+3?", ground_truth="8")]

    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{8}",
                     log_probs=[-0.1, -0.2], answer="8", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[3, 4], text="\\boxed{7}",
                     log_probs=[-0.3, -0.4], answer="7", correct=False, reward=0.0)],
    }

    alice_disc = Rollout(problem_idx=0, player_id=0, tokens=[10, 11],
                         text="EVALUATION: I'm sure.\nFINAL ANSWER: \\boxed{8}",
                         log_probs=[-0.1, -0.2])
    bob_disc = Rollout(problem_idx=0, player_id=1, tokens=[12, 13],
                       text="EVALUATION: You're right.\nFINAL ANSWER: \\boxed{8}",
                       log_probs=[-0.3, -0.4])

    def fake_generate(vllm, prompt_token_ids, problem_indices, player_id, max_new_tokens):
        return [alice_disc] if player_id == 0 else [bob_disc]

    with patch("crisp.workflow.discussion_step.generate_samples", side_effect=fake_generate):
        disc_results, majority_answers = run_discussion(ctx, rollouts, problems)

    all_results = []
    for pid in disc_results:
        all_results.extend(disc_results[pid])
    assert len(all_results) == 2

    for dr in all_results:
        assert isinstance(dr, DiscussionResult)
        assert dr.final_answer == "8"
        assert dr.correct is True

    # Post-discussion reward: 1.0 if correct, 0.0 otherwise
    for dr in all_results:
        if dr.correct:
            assert dr.reward == 1.0, f"Correct discussion result should have reward=1.0, got {dr.reward}"
        else:
            assert dr.reward == 0.0


def test_run_discussion_multiple_problems():
    """Discussion only triggers for problems with disagreement."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [
        Problem(text="2+2?", ground_truth="4"),
        Problem(text="3+3?", ground_truth="6"),
    ]

    rollouts = {
        0: [
            Rollout(problem_idx=0, player_id=0, tokens=[1], text="\\boxed{4}",
                    log_probs=[-0.1], answer="4", correct=True, reward=1.0),
            Rollout(problem_idx=1, player_id=0, tokens=[2], text="\\boxed{5}",
                    log_probs=[-0.2], answer="5", correct=False, reward=0.0),
        ],
        1: [
            Rollout(problem_idx=0, player_id=1, tokens=[3], text="\\boxed{4}",
                    log_probs=[-0.3], answer="4", correct=True, reward=1.0),
            Rollout(problem_idx=1, player_id=1, tokens=[4], text="\\boxed{6}",
                    log_probs=[-0.4], answer="6", correct=True, reward=1.0),
        ],
    }

    alice_disc = Rollout(problem_idx=1, player_id=0, tokens=[10],
                         text="EVALUATION: ok\nFINAL ANSWER: \\boxed{6}",
                         log_probs=[-0.1])
    bob_disc = Rollout(problem_idx=1, player_id=1, tokens=[11],
                       text="EVALUATION: yes\nFINAL ANSWER: \\boxed{6}",
                       log_probs=[-0.2])

    def fake_generate(vllm, prompt_token_ids, problem_indices, player_id, max_new_tokens):
        return [alice_disc] if player_id == 0 else [bob_disc]

    with patch("crisp.workflow.discussion_step.generate_samples", side_effect=fake_generate):
        disc_results, majority_answers = run_discussion(ctx, rollouts, problems)

    all_results = []
    for pid in disc_results:
        all_results.extend(disc_results[pid])
    assert len(all_results) == 2
    assert all(dr.problem_idx == 1 for dr in all_results)


def test_run_discussion_overlong_penalty():
    """Overlong discussion responses get penalty subtracted from reward."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [Problem(text="Hard problem", ground_truth="42")]

    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{42}",
                     log_probs=[-0.1, -0.2], answer="42", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[3, 4], text="\\boxed{7}",
                     log_probs=[-0.3, -0.4], answer="7", correct=False, reward=0.0)],
    }

    # Create a discussion rollout with tokens exceeding post_discussion_l_max (4096)
    long_tokens = list(range(5000))
    alice_disc = Rollout(problem_idx=0, player_id=0, tokens=long_tokens,
                         text="EVALUATION: ok\nFINAL ANSWER: \\boxed{42}",
                         log_probs=[0.0] * 5000)
    bob_disc = Rollout(problem_idx=0, player_id=1, tokens=[12, 13],
                       text="EVALUATION: right\nFINAL ANSWER: \\boxed{42}",
                       log_probs=[-0.3, -0.4])

    def fake_generate(vllm, prompt_token_ids, problem_indices, player_id, max_new_tokens):
        return [alice_disc] if player_id == 0 else [bob_disc]

    with patch("crisp.workflow.discussion_step.generate_samples", side_effect=fake_generate):
        disc_results, _ = run_discussion(ctx, rollouts, problems)

    # Player 0's discussion result should have reduced reward due to overlong penalty
    p0_results = disc_results[0]
    assert len(p0_results) == 1
    assert p0_results[0].correct is True
    # reward = 1.0 (correct) - overlong_penalty(5000, 4096, 1024) ≈ 0.88
    assert p0_results[0].reward < 1.0
    assert p0_results[0].reward > 0.0

    # Player 1's short discussion result should have full reward
    p1_results = disc_results[1]
    assert p1_results[0].reward == 1.0


def test_run_discussion_syncs_weights_per_player():
    """Weight sync happens per-player: Alice's weights before her generation,
    Bob's weights before his generation."""
    from crisp.workflow.discussion_step import run_discussion

    ctx = _make_ctx()
    problems = [Problem(text="What is 5+3?", ground_truth="8")]

    rollouts = {
        0: [Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="\\boxed{8}",
                     log_probs=[-0.1, -0.2], answer="8", correct=True, reward=1.0)],
        1: [Rollout(problem_idx=0, player_id=1, tokens=[3, 4], text="\\boxed{7}",
                     log_probs=[-0.3, -0.4], answer="7", correct=False, reward=0.0)],
    }

    # Track call order to verify sync happens before each player's generation
    call_order = []

    def track_sync_alice(*args, **kwargs):
        call_order.append("sync_alice")

    def track_sync_bob(*args, **kwargs):
        call_order.append("sync_bob")

    ctx.ds_alice.sync_weights = track_sync_alice
    ctx.ds_bob.sync_weights = track_sync_bob

    alice_disc = Rollout(problem_idx=0, player_id=0, tokens=[10, 11],
                         text="EVALUATION: I'm sure.\nFINAL ANSWER: \\boxed{8}",
                         log_probs=[-0.1, -0.2])
    bob_disc = Rollout(problem_idx=0, player_id=1, tokens=[12, 13],
                       text="EVALUATION: You're right.\nFINAL ANSWER: \\boxed{8}",
                       log_probs=[-0.3, -0.4])

    def fake_generate(vllm, prompt_token_ids, problem_indices, player_id, max_new_tokens):
        call_order.append(f"generate_player_{player_id}")
        if player_id == 0:
            return [alice_disc]
        else:
            return [bob_disc]

    with patch("crisp.workflow.discussion_step.generate_samples", side_effect=fake_generate):
        disc_results, _ = run_discussion(ctx, rollouts, problems)

    # Verify both syncs happened
    assert "sync_alice" in call_order, f"Alice sync not called. Order: {call_order}"
    assert "sync_bob" in call_order, f"Bob sync not called. Order: {call_order}"

    # Verify ordering: sync before generate for each player
    alice_sync_idx = call_order.index("sync_alice")
    alice_gen_idx = call_order.index("generate_player_0")
    assert alice_sync_idx < alice_gen_idx, (
        f"Alice sync must happen before Alice generate. Order: {call_order}"
    )

    bob_sync_idx = call_order.index("sync_bob")
    bob_gen_idx = call_order.index("generate_player_1")
    assert bob_sync_idx < bob_gen_idx, (
        f"Bob sync must happen before Bob generate. Order: {call_order}"
    )

    # Verify results are still correct
    all_results = []
    for pid in disc_results:
        all_results.extend(disc_results[pid])
    assert len(all_results) == 2
