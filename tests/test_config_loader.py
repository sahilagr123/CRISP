"""Tests for YAML config loader."""
from crisp.config import CRISPConfig, InfraConfig, TrainingConfig
from crisp.config_loader import load_config


def test_load_empty_yaml(tmp_path):
    """Empty YAML produces default CRISPConfig."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    config = load_config(p)
    assert isinstance(config, CRISPConfig)
    assert config.infra.learning_rate == 1e-6
    assert config.training.model_name == "Qwen/Qwen3-4B-Instruct-2507"


def test_load_partial_yaml(tmp_path):
    """YAML with only infra section fills infra, others default."""
    p = tmp_path / "partial.yaml"
    p.write_text("infra:\n  learning_rate: 0.001\n")
    config = load_config(p)
    assert config.infra.learning_rate == 0.001
    assert config.player.rollouts_per_problem == 8  # default


def test_load_full_yaml(tmp_path):
    """YAML with all sections produces correct config."""
    p = tmp_path / "full.yaml"
    p.write_text(
        "player:\n  rollouts_per_problem: 4\n"
        "coach:\n  update_freq: 3\n"
        "advantage:\n  epsilon: 0.000001\n"
        "grpo:\n  dcpo_alpha: 2.0\n"
        "infra:\n  num_gpus_per_node: 2\n"
        "training:\n  num_iterations: 50\n"
    )
    config = load_config(p)
    assert config.player.rollouts_per_problem == 4
    assert config.coach.update_freq == 3
    assert config.advantage.epsilon == 0.000001
    assert config.grpo.dcpo_alpha == 2.0
    assert config.infra.num_gpus_per_node == 2
    assert config.training.num_iterations == 50


def test_overrides_apply(tmp_path):
    """Dot-path overrides modify loaded config."""
    p = tmp_path / "base.yaml"
    p.write_text("infra:\n  learning_rate: 0.01\n")
    config = load_config(p, overrides={"infra.learning_rate": 1e-4})
    assert config.infra.learning_rate == 1e-4


def test_override_creates_missing_section(tmp_path):
    """Override for missing section creates the section."""
    p = tmp_path / "empty.yaml"
    p.write_text("")
    config = load_config(p, overrides={"training.num_iterations": 50})
    assert config.training.num_iterations == 50


def test_training_config_defaults():
    """TrainingConfig() has correct defaults."""
    tc = TrainingConfig()
    assert tc.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert tc.num_iterations == 3000
    assert tc.checkpoint_dir == "checkpoints"
    assert tc.save_freq == 10
    assert tc.log_freq == 1
    assert tc.attn_implementation == "eager"
    assert tc.ref_reward_offload is False
