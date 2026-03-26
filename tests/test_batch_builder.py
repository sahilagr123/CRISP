"""Tests for batch_builder — filter_dynamic_sampling, build_player_batch, build_coach_batch."""
from crisp.types import Rollout, DiscussionResult, Problem, TokenSequence, TrainingBatch


def test_filter_dynamic_sampling_removes_zero_variance():
    """filter_dynamic_sampling removes problems where all rollouts have same reward."""
    from crisp.training.batch_builder import filter_dynamic_sampling

    rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1], text="a",
                log_probs=[-0.1], answer="1", correct=True, reward=1.0),
        Rollout(problem_idx=0, player_id=0, tokens=[2], text="b",
                log_probs=[-0.2], answer="1", correct=True, reward=1.0),
        Rollout(problem_idx=1, player_id=0, tokens=[3], text="c",
                log_probs=[-0.3], answer="2", correct=True, reward=1.0),
        Rollout(problem_idx=1, player_id=0, tokens=[4], text="d",
                log_probs=[-0.4], answer="3", correct=False, reward=0.0),
    ]
    result = filter_dynamic_sampling(rollouts)
    assert len(result) == 2
    assert all(r.problem_idx == 1 for r in result)


def test_build_player_batch_basic():
    """build_player_batch creates a TrainingBatch from rollouts and advantages."""
    from crisp.training.batch_builder import build_player_batch

    rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2, 3], text="a",
                log_probs=[-0.1, -0.2, -0.3], answer="1", correct=True, reward=1.0),
        Rollout(problem_idx=0, player_id=0, tokens=[4, 5], text="b",
                log_probs=[-0.4, -0.5], answer="2", correct=False, reward=0.0),
    ]
    advantages = [0.5, -0.5]

    batch = build_player_batch(rollouts, advantages)
    assert isinstance(batch, TrainingBatch)
    assert len(batch.sequences) == 2
    assert len(batch.advantages) == 2
    assert batch.advantages == [0.5, -0.5]
    assert batch.is_post_discussion == [False, False]


def test_build_player_batch_with_discussion():
    """build_player_batch handles mixed pre/post-discussion sequences."""
    from crisp.training.batch_builder import build_player_batch

    rollouts = [
        Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text="a",
                log_probs=[-0.1, -0.2], answer="1", correct=True, reward=1.0),
    ]
    discussion_results = [
        DiscussionResult(problem_idx=0, player_id=0, tokens=[3, 4, 5], text="b",
                         log_probs=[-0.3, -0.4, -0.5], evaluation_text="eval",
                         final_answer="1", correct=True, reward=1.3),
    ]
    pre_advantages = [0.5]
    post_advantages = [0.8]

    batch = build_player_batch(
        rollouts, pre_advantages,
        discussion_results=discussion_results,
        post_advantages=post_advantages,
    )
    assert len(batch.sequences) == 2
    assert batch.is_post_discussion == [False, True]
    assert batch.advantages == [0.5, 0.8]


def test_build_player_batch_empty():
    """build_player_batch with empty input returns empty batch."""
    from crisp.training.batch_builder import build_player_batch

    batch = build_player_batch([], [])
    assert len(batch.sequences) == 0
    assert len(batch.advantages) == 0


def test_build_coach_batch_basic():
    """build_coach_batch creates a TrainingBatch from problems and advantages."""
    from crisp.training.batch_builder import build_coach_batch

    problems = [
        Problem(text="What is 2+2?", ground_truth="4",
                coach_sequence=TokenSequence(tokens=[10, 20, 30],
                                             log_probs=[-0.1, -0.2, -0.3])),
        Problem(text="What is 3+3?", ground_truth="6",
                coach_sequence=TokenSequence(tokens=[40, 50],
                                             log_probs=[-0.4, -0.5])),
    ]
    advantages = [0.3, -0.3]

    batch = build_coach_batch(problems, advantages)
    assert isinstance(batch, TrainingBatch)
    assert len(batch.sequences) == 2
    assert batch.advantages == [0.3, -0.3]
    assert batch.sequences[0].tokens == [10, 20, 30]
    assert batch.is_post_discussion == [False, False]


def test_build_coach_batch_skips_no_sequence():
    """build_coach_batch skips problems without coach_sequence."""
    from crisp.training.batch_builder import build_coach_batch

    problems = [
        Problem(text="Q1", ground_truth="1", coach_sequence=None),
        Problem(text="Q2", ground_truth="2",
                coach_sequence=TokenSequence(tokens=[1, 2], log_probs=[-0.1, -0.2])),
    ]
    advantages = [0.5, -0.5]

    batch = build_coach_batch(problems, advantages)
    assert len(batch.sequences) == 1
    assert batch.advantages == [-0.5]
