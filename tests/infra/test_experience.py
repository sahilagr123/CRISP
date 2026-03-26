"""Tests for experience.py — vLLM output to CRISP Rollout mapping."""
import sys
from unittest.mock import MagicMock, patch
import torch


def _make_mock_vllm_output(prompt_token_ids, output_token_ids, logprobs=None, text="generated text"):
    """Helper to create a mock vLLM CompletionOutput."""
    output = MagicMock()
    output.prompt_token_ids = prompt_token_ids
    mock_completion = MagicMock()
    mock_completion.token_ids = output_token_ids
    mock_completion.text = text
    if logprobs is not None:
        mock_logprobs = []
        for tid, lp in zip(output_token_ids, logprobs):
            mock_entry = MagicMock()
            mock_entry.logprob = lp
            mock_logprobs.append({tid: mock_entry})
        mock_completion.logprobs = mock_logprobs
    else:
        mock_completion.logprobs = None
    output.outputs = [mock_completion]
    return output


def test_map_vllm_output_to_rollout():
    """Single vLLM output maps to a Rollout with correct fields."""
    from crisp.infra.experience import map_vllm_output_to_rollout
    output = _make_mock_vllm_output(
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[10, 20, 30],
    )
    rollout = map_vllm_output_to_rollout(output, problem_idx=0, player_id=1)
    assert rollout.problem_idx == 0
    assert rollout.player_id == 1
    assert rollout.tokens == [1, 2, 3, 10, 20, 30]
    assert len(rollout.log_probs) == 6
    assert rollout.text == "generated text"
    assert rollout.reward == 0.0


def test_map_vllm_output_with_logprobs():
    """When vLLM returns logprobs, they're captured in the Rollout."""
    from crisp.infra.experience import map_vllm_output_to_rollout
    output = _make_mock_vllm_output(
        prompt_token_ids=[1, 2],
        output_token_ids=[10, 20],
        logprobs=[-0.5, -1.2],
    )
    rollout = map_vllm_output_to_rollout(output, problem_idx=0, player_id=0)
    assert rollout.log_probs[:2] == [0.0, 0.0]
    assert abs(rollout.log_probs[2] - (-0.5)) < 1e-6
    assert abs(rollout.log_probs[3] - (-1.2)) < 1e-6


def test_generate_samples_distributes_across_engines():
    """generate_samples distributes prompts across vLLM engines evenly."""
    from crisp.infra.experience import generate_samples

    engine1 = MagicMock()
    engine2 = MagicMock()

    output1 = _make_mock_vllm_output([1, 2], [10, 20])
    output2 = _make_mock_vllm_output([3, 4], [30, 40])

    engine1.generate.remote.return_value = "gen_ref1"
    engine2.generate.remote.return_value = "gen_ref2"

    # Mock ray and vllm modules so deferred imports inside generate_samples work.
    mock_ray = MagicMock()
    mock_ray.get.return_value = [[output1], [output2]]
    mock_vllm = MagicMock()

    with patch.dict(sys.modules, {"ray": mock_ray, "vllm": mock_vllm}):
        rollouts = generate_samples(
            vllm_engines=[engine1, engine2],
            prompt_token_ids=[[1, 2], [3, 4]],
            problem_indices=[0, 1],
            player_id=0,
            max_new_tokens=1024,
        )
        assert len(rollouts) == 2
        assert rollouts[0].problem_idx == 0
        assert rollouts[1].problem_idx == 1
