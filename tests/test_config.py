"""Tests for configuration defaults and validation."""
import pytest

from crisp.config import CRISPConfig, PlayerConfig, CoachConfig, AdvantageConfig, GRPOConfig


class TestCRISPConfig:
    def test_defaults_match_spec(self):
        cfg = CRISPConfig()
        assert cfg.player.rollouts_per_problem == 8
        assert cfg.player.solve_reward == 1.0
        assert cfg.player.wrong_reward == 0.0
        assert cfg.player.no_box_penalty == -0.5
        assert cfg.player.persuader_bonus == 0.3
        assert cfg.coach.batch_size == 8
        assert cfg.coach.discussion_alpha == 0.3
        assert cfg.coach.repetition_lambda == 1.0
        assert cfg.coach.repetition_tau == 0.85
        assert cfg.coach.repetition_window == 10
        assert cfg.coach.embedding_dim == 384
        assert cfg.advantage.ema_eta == 0.2
        assert cfg.advantage.ema_init_mu == 0.5
        assert cfg.advantage.ema_init_sigma_sq == 0.25
        assert cfg.grpo.dcpo_alpha == 3.0
        assert cfg.grpo.clip_low == 0.2
        assert cfg.grpo.clip_high == 0.28
        assert cfg.grpo.js_beta == 0.005
        assert cfg.grpo.pre_discussion_l_max == 8192
        assert cfg.grpo.pre_discussion_buffer == 2048
        assert cfg.grpo.post_discussion_l_max == 4096
        assert cfg.grpo.post_discussion_buffer == 1024

    def test_custom_config(self):
        cfg = CRISPConfig(player=PlayerConfig(rollouts_per_problem=4))
        assert cfg.player.rollouts_per_problem == 4
        assert cfg.coach.batch_size == 8  # Other defaults preserved


def test_infra_config_defaults():
    """InfraConfig provides sensible defaults for single-GPU dev."""
    from crisp.config import InfraConfig

    cfg = InfraConfig()
    assert cfg.zero_stage == 2
    assert cfg.bf16 is True
    assert cfg.num_gpus_per_node == 1
    assert cfg.num_nodes == 1
    assert cfg.vllm_tensor_parallel_size == 1
    assert cfg.vllm_num_engines == 1
    assert cfg.vllm_gpu_memory_utilization == 0.85
    assert cfg.vllm_enable_sleep is True
    assert cfg.adam_offload is False
    assert cfg.gradient_checkpointing is True
    assert cfg.max_model_len == 10240
    assert cfg.micro_train_batch_size == 1
    assert cfg.lora_rank == 0  # 0 means full fine-tuning
    assert cfg.lora_alpha == 16
    assert cfg.seed == 42


def test_infra_config_in_crisp_config():
    """CRISPConfig includes infra sub-config."""
    from crisp.config import CRISPConfig

    cfg = CRISPConfig()
    assert cfg.infra is not None
    assert cfg.infra.zero_stage == 2


def test_infra_config_custom():
    """InfraConfig accepts overrides."""
    from crisp.config import InfraConfig

    cfg = InfraConfig(num_nodes=4, num_gpus_per_node=8, zero_stage=3, lora_rank=64)
    assert cfg.num_nodes == 4
    assert cfg.num_gpus_per_node == 8
    assert cfg.zero_stage == 3
    assert cfg.lora_rank == 64


def test_coach_config_update_freq():
    """CoachConfig has update_freq with default 1."""
    from crisp.config import CoachConfig
    cfg = CoachConfig()
    assert cfg.update_freq == 1


def test_coach_config_templates():
    """CoachConfig has discussion and coach prompt templates."""
    from crisp.config import CoachConfig
    cfg = CoachConfig()
    assert isinstance(cfg.discussion_template, str)
    assert "{problem}" in cfg.discussion_template
    assert "{own_solution}" in cfg.discussion_template
    assert "{other_solution}" in cfg.discussion_template
    assert isinstance(cfg.coach_prompt_template, str)


def test_training_config_defaults():
    """TrainingConfig has correct model name defaults."""
    from crisp.config import TrainingConfig
    tc = TrainingConfig()
    assert tc.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert tc.coach_model_name == "Qwen/Qwen3-14B"


def test_training_config_coach_model_none_fallback():
    """When coach_model_name is None, it should be treated as model_name."""
    from crisp.config import TrainingConfig
    tc = TrainingConfig(coach_model_name=None)
    assert tc.coach_model_name is None
