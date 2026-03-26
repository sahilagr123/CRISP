"""LoRA adapter save and merge utilities for CRISP.

save_lora_adapters: Non-destructive save of adapter weights from live model.
merge_and_save: Load adapters from disk, merge into base model, save full model.
                Never touches the live training model.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import torch


def has_lora(strategy: Any) -> bool:
    """Check if the model has LoRA adapters (is a PeftModel)."""
    try:
        from peft import PeftModel
        return isinstance(strategy.module, PeftModel)
    except (ImportError, AttributeError):
        return False


def save_lora_adapters(strategy: Any, path: str) -> None:
    """Save LoRA adapter weights only. Non-destructive.

    Saves adapter_model.safetensors + adapter_config.json to path.
    """
    os.makedirs(path, exist_ok=True)
    model = strategy.module
    model.save_pretrained(path)


def merge_and_save(
    adapter_path: str,
    output_path: str,
    base_model_name: str,
    tokenizer_name: Optional[str] = None,
) -> None:
    """Load adapters from disk, merge into base model, save full model.

    NEVER operates on the live training model. Safe to call during or
    after training. Can run in a separate process.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)

    if tokenizer_name:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        tok.save_pretrained(output_path)
