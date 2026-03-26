"""Tests for StepCollector and IterationData."""
from crisp.workflow.collector import IterationData, StepCollector
from crisp.workflow.context import StepResult


def test_collector_starts_empty():
    collector = StepCollector()
    assert len(collector.iterations) == 0


def test_collector_records_iteration():
    collector = StepCollector()
    data = IterationData(
        iteration=0,
        problems=[],
        rollouts={0: [], 1: []},
        majority_answers={},
        discussion_results={0: [], 1: []},
        player_loss=0.5,
        coach_loss=0.1,
        coach_rewards=[0.5, 0.3],
        result=StepResult(
            alice_loss=0.5, bob_loss=0.5, coach_loss=0.1, num_problems=2,
            num_discussions=1, player_accuracy=0.75, coach_iteration=True,
        ),
    )
    collector.record(data)
    assert len(collector.iterations) == 1
    assert collector.iterations[0].iteration == 0
    assert collector.iterations[0].coach_rewards == [0.5, 0.3]


def test_collector_accumulates_multiple():
    collector = StepCollector()
    for i in range(5):
        data = IterationData(
            iteration=i, problems=[], rollouts={0: [], 1: []},
            majority_answers={}, discussion_results={0: [], 1: []},
            player_loss=float(i), coach_loss=None, coach_rewards=None,
            result=StepResult(
                alice_loss=float(i), bob_loss=float(i), coach_loss=None, num_problems=0,
                num_discussions=0, player_accuracy=0.0, coach_iteration=False,
            ),
        )
        collector.record(data)
    assert len(collector.iterations) == 5
    assert [d.iteration for d in collector.iterations] == [0, 1, 2, 3, 4]
