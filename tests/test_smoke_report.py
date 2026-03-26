"""Tests for the smoke test report generator."""
import numpy as np

from crisp.config import CRISPConfig
from crisp.types import DiscussionResult, Problem, Rollout
from crisp.workflow.collector import IterationData, StepCollector
from crisp.workflow.context import StepResult


def _make_collector_with_data():
    """Build a StepCollector with 2 synthetic iterations."""
    collector = StepCollector()

    for i in range(2):
        problems = [
            Problem(text=f"What is {i}+1?", ground_truth=str(i + 1),
                    coach_embedding=np.zeros(384, dtype=np.float32)),
            Problem(text=f"What is {i}+2?", ground_truth=str(i + 2),
                    coach_embedding=np.zeros(384, dtype=np.float32)),
        ]
        rollouts = {
            0: [
                Rollout(problem_idx=0, player_id=0, tokens=[1, 2], text=f"\\boxed{{{i+1}}}",
                        log_probs=[-0.1, -0.2], answer=str(i + 1), correct=True, reward=1.0),
                Rollout(problem_idx=1, player_id=0, tokens=[3, 4], text="\\boxed{99}",
                        log_probs=[-0.3, -0.4], answer="99", correct=False, reward=0.0),
            ],
            1: [
                Rollout(problem_idx=0, player_id=1, tokens=[5, 6], text="\\boxed{99}",
                        log_probs=[-0.5, -0.6], answer="99", correct=False, reward=0.0),
                Rollout(problem_idx=1, player_id=1, tokens=[7, 8], text=f"\\boxed{{{i+2}}}",
                        log_probs=[-0.7, -0.8], answer=str(i + 2), correct=True, reward=1.0),
            ],
        }
        disc_results = {
            0: [DiscussionResult(
                problem_idx=0, player_id=0, tokens=[9], text="EVALUATION: checking\nFINAL ANSWER: \\boxed{1}",
                log_probs=[-0.1], evaluation_text="checking", final_answer=str(i + 1),
                correct=True, reward=1.0,
            )],
            1: [DiscussionResult(
                problem_idx=0, player_id=1, tokens=[10], text="EVALUATION: ok\nFINAL ANSWER: \\boxed{1}",
                log_probs=[-0.2], evaluation_text="ok", final_answer=str(i + 1),
                correct=True, reward=1.0,
            )],
        }
        majority = {(0, 0): str(i + 1), (1, 0): "99", (0, 1): "99", (1, 1): str(i + 2)}

        alice_loss = 0.5 - i * 0.1
        bob_loss = 0.4 - i * 0.1
        result = StepResult(
            alice_loss=alice_loss, bob_loss=bob_loss,
            coach_loss=0.3 - i * 0.05,
            num_problems=2, num_discussions=1,
            player_accuracy=0.5, coach_iteration=True,
        )
        collector.record(IterationData(
            iteration=i, problems=problems, rollouts=rollouts,
            majority_answers=majority, discussion_results=disc_results,
            player_loss=(alice_loss + bob_loss) / 2, coach_loss=result.coach_loss,
            coach_rewards=[0.75, 0.50], result=result,
        ))
    return collector


def test_report_generates_markdown():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    config = CRISPConfig()
    report = generate_report(collector, config)

    assert isinstance(report, str)
    assert "# CRISP Smoke Test Report" in report
    assert "## Summary" in report


def test_report_contains_problems():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "What is 0+1?" in report
    assert "Ground Truth" in report


def test_report_contains_rollout_table():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "| Player" in report
    assert "Alice" in report
    assert "Bob" in report


def test_report_contains_discussion():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "Discussion" in report
    assert "DISAGREE" in report or "Disagree" in report or "disagree" in report


def test_report_contains_summary_table():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    # Summary table should have iteration numbers
    assert "| 0" in report
    assert "| 1" in report


def test_report_contains_coach_rewards():
    from scripts.write_smoke_report import generate_report
    collector = _make_collector_with_data()
    report = generate_report(collector, CRISPConfig())

    assert "Coach Rewards" in report
    assert "0.75" in report
