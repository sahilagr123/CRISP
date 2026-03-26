"""Tests for LoRA adapter save and merge utilities."""
import sys
from unittest.mock import MagicMock, patch


class _FakePeftModel:
    """Stand-in for peft.PeftModel in isinstance checks."""
    pass


def test_has_lora_true():
    """has_lora returns True for a PeftModel."""
    from crisp.infra.lora_utils import has_lora

    mock_peft = MagicMock()
    mock_peft.PeftModel = _FakePeftModel
    strategy = MagicMock()
    strategy.module = _FakePeftModel()

    with patch.dict(sys.modules, {"peft": mock_peft}):
        assert has_lora(strategy) is True


def test_has_lora_false():
    """has_lora returns False for a non-PEFT model."""
    from crisp.infra.lora_utils import has_lora

    mock_peft = MagicMock()
    mock_peft.PeftModel = _FakePeftModel
    strategy = MagicMock()
    # strategy.module is a MagicMock, not a _FakePeftModel
    strategy.module = MagicMock()

    with patch.dict(sys.modules, {"peft": mock_peft}):
        assert has_lora(strategy) is False


def test_has_lora_no_peft():
    """has_lora returns False when peft is not installed."""
    from crisp.infra.lora_utils import has_lora

    strategy = MagicMock()
    with patch.dict(sys.modules, {"peft": None}):
        assert has_lora(strategy) is False


def test_save_lora_adapters():
    """save_lora_adapters calls model.save_pretrained."""
    from crisp.infra.lora_utils import save_lora_adapters

    strategy = MagicMock()
    mock_model = MagicMock()
    strategy.module = mock_model

    with patch("crisp.infra.lora_utils.os.makedirs"):
        save_lora_adapters(strategy, "/tmp/adapters")

    mock_model.save_pretrained.assert_called_once_with("/tmp/adapters")


def test_merge_and_save_loads_from_disk():
    """merge_and_save loads adapters from disk, merges, saves."""
    from crisp.infra.lora_utils import merge_and_save

    mock_peft = MagicMock()
    mock_transformers = MagicMock()
    mock_base_model = MagicMock()
    mock_peft_model = MagicMock()
    mock_merged = MagicMock()

    mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_base_model
    mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model
    mock_peft_model.merge_and_unload.return_value = mock_merged

    with patch.dict(sys.modules, {"peft": mock_peft, "transformers": mock_transformers}):
        merge_and_save("/tmp/adapters", "/tmp/merged", "base-model")

    mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
    mock_peft.PeftModel.from_pretrained.assert_called_once_with(mock_base_model, "/tmp/adapters")
    mock_peft_model.merge_and_unload.assert_called_once()
    mock_merged.save_pretrained.assert_called_once_with("/tmp/merged")


def test_merge_and_save_with_tokenizer():
    """merge_and_save saves tokenizer when tokenizer_name provided."""
    from crisp.infra.lora_utils import merge_and_save

    mock_peft = MagicMock()
    mock_transformers = MagicMock()
    mock_peft_model = MagicMock()
    mock_merged = MagicMock()
    mock_tokenizer = MagicMock()

    mock_peft.PeftModel.from_pretrained.return_value = mock_peft_model
    mock_peft_model.merge_and_unload.return_value = mock_merged
    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

    with patch.dict(sys.modules, {"peft": mock_peft, "transformers": mock_transformers}):
        merge_and_save("/tmp/adapters", "/tmp/merged", "base-model", tokenizer_name="base-model")

    mock_transformers.AutoTokenizer.from_pretrained.assert_called_once_with(
        "base-model", trust_remote_code=True,
    )
    mock_tokenizer.save_pretrained.assert_called_once_with("/tmp/merged")
