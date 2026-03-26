"""Actor model wrapper for HuggingFace CausalLM with LoRA support.

Vendored from MARTI (https://github.com/TsinghuaC3I/MARTI).
Provides model loading, forward pass (log-probs), and gradient checkpointing.
Ring attention removed (not needed for CRISP).
"""
from typing import Optional

try:
    import deepspeed
except ImportError:
    deepspeed = None

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

# Chunk size for lm_head computation along the sequence dimension.
# For V=151936, each chunk uses [B, chunk, V] × fp32 ≈ 0.59 GiB.
# Without chunking, L=10K would need [B, 10K, V] × fp32 ≈ 5.7 GiB,
# plus ~5.7 GiB for backward intermediates — easily causing OOM.
LM_HEAD_CHUNK_SIZE = 1024


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Compute per-token log probabilities from logits.

    Uses logsumexp trick to avoid materializing the full [B, T, V]
    log_softmax tensor. For V=151K this saves ~4.7 GiB per chunk.

    Args:
        logits: [B, T, V] model output logits
        labels: [B, T] token IDs to evaluate
        temperature: softmax temperature

    Returns:
        [B, T] log probabilities for each label token
    """
    if temperature != 1.0:
        logits = logits / temperature
    # log_softmax(x)_i = x_i - logsumexp(x)
    # Gather first to avoid creating full [B, T, V] log_softmax output
    selected = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
    lse = logits.logsumexp(dim=-1)  # [B, T]
    return selected - lse


def _lm_head_chunk_fn(hidden_chunk, labels_chunk, weight, bias, temperature):
    """Compute log-probs for a single sequence chunk (checkpoint-compatible).

    Computes lm_head(hidden) → logits → log_probs in one function so that
    torch.utils.checkpoint can discard the intermediate [B, chunk, V] logits
    during forward and recompute them during backward.
    """
    logits = F.linear(hidden_chunk, weight, bias).to(torch.float32)
    return log_probs_from_logits(logits, labels_chunk, temperature)


def _get_causal_lm(model):
    """Navigate through PEFT wrapper to get the base CausalLM."""
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            return model.get_base_model()
    except ImportError:
        pass
    return model


def chunked_lm_head_log_probs(hidden_states, lm_head, labels, temperature=1.0):
    """Compute log-probs by chunking the lm_head along the sequence dimension.

    Instead of materializing the full [B, L, V] logits tensor (up to 5.7 GiB
    for L=10K, V=152K in fp32), processes LM_HEAD_CHUNK_SIZE tokens at a time.
    Each chunk is wrapped in torch.utils.checkpoint during training so that
    the intermediate [B, chunk, V] logits are discarded after forward and
    recomputed during backward.

    Peak lm_head memory: O(chunk × V) ≈ 0.6 GiB instead of O(L × V) ≈ 5.7 GiB.
    """
    L = hidden_states.shape[1]
    training = hidden_states.requires_grad
    weight = lm_head.weight
    bias = lm_head.bias if hasattr(lm_head, 'bias') else None

    # Short sequence without grad: compute directly (no overhead)
    if L <= LM_HEAD_CHUNK_SIZE and not training:
        logits = lm_head(hidden_states).to(torch.float32)
        return log_probs_from_logits(logits, labels, temperature)

    chunks = []
    for start in range(0, L, LM_HEAD_CHUNK_SIZE):
        end = min(start + LM_HEAD_CHUNK_SIZE, L)
        chunk_h = hidden_states[:, start:end, :]
        chunk_labels = labels[:, start:end]

        if training:
            chunk_lp = torch_checkpoint(
                _lm_head_chunk_fn,
                chunk_h, chunk_labels, weight, bias, temperature,
                use_reentrant=False,
            )
        else:
            logits = lm_head(chunk_h).to(torch.float32)
            chunk_lp = log_probs_from_logits(logits, chunk_labels, temperature)
            del logits

        chunks.append(chunk_lp)

    return torch.cat(chunks, dim=1)


class Actor(nn.Module):
    """HuggingFace CausalLM wrapper with LoRA and DeepSpeed support.

    Can be initialized from a pretrained model path (str) or an existing nn.Module.
    """

    def __init__(
        self,
        pretrain_or_model,
        bf16=True,
        load_in_4bit=False,
        lora_rank=0,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=None,
        ds_config=None,
        device_map=None,
        attn_implementation="flash_attention_2",
        temperature=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        if isinstance(pretrain_or_model, str):
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None

            if load_in_4bit:
                assert bf16, "4-bit quantization requires bf16"
                nf4_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                nf4_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )

            # LoRA
            if lora_rank > 0:
                from peft import LoraConfig, TaskType, get_peft_model
                from peft.tuners.lora import LoraLayer

                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)

            # MoE support
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                self.model.config.output_router_logits = True
                if deepspeed is not None:
                    for m in self.model.modules():
                        if "SparseMoeBlock" in m.__class__.__name__:
                            deepspeed.utils.set_z3_leaf_modules(self.model, [m.__class__])
                            break

            self.model.config.use_cache = False
        else:
            self.model = pretrain_or_model

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
    ) -> torch.Tensor:
        """Compute action log-probs for given sequences.

        Uses a chunked lm_head to avoid materializing the full [B, L, V]
        logits tensor. For V=152K, a 10K-token sequence would need ~5.7 GiB
        in fp32 just for logits, plus ~5.7 GiB for backward intermediates.
        Chunking reduces this to ~0.6 GiB peak.

        Args:
            sequences: [B, S] input token IDs
            action_mask: [B, A] mask for action tokens (response portion)
            attention_mask: [B, S] attention mask

        Returns:
            [B, A] action log-probabilities (if action_mask provided)
            or output dict (if return_output=True and no action_mask)
        """
        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # Split into backbone (transformer) + lm_head so we can chunk the
        # lm_head along the sequence dimension for memory efficiency.
        causal_lm = _get_causal_lm(self.model)
        backbone = causal_lm.model   # transformer layers
        lm_head = causal_lm.lm_head  # [H] -> [V] projection

        # One-time GC diagnostic: log backbone state on first training forward
        if not hasattr(self, '_gc_checked') and sequences.requires_grad is False:
            import logging as _fwd_log
            _l = _fwd_log.getLogger(__name__)
            gc_val = getattr(backbone, 'gradient_checkpointing', None)
            gc_func = getattr(backbone, '_gradient_checkpointing_func', None)
            _l.info(
                "Forward GC check: backbone.gradient_checkpointing=%s, "
                "has_gc_func=%s, backbone.training=%s, L=%d",
                gc_val, gc_func is not None, backbone.training, sequences.shape[1],
            )
            self._gc_checked = True

        # Get hidden states from transformer (gradient checkpointing applied here)
        backbone_output = backbone(
            sequences, attention_mask=attention_mask, position_ids=position_ids,
        )
        hidden_states = backbone_output[0]  # [B, L, H]

        if action_mask is None and return_output:
            output = {"logits": lm_head(hidden_states).to(torch.float32)}
            return output

        # Compute log-probs via chunked lm_head (each chunk checkpointed)
        log_probs = chunked_lm_head_log_probs(
            hidden_states, lm_head, rolled_sequences, self.temperature,
        )
        log_probs = log_probs[:, :-1]

        if action_mask is None:
            return log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1]:] * action_mask.float()
        return action_log_probs

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        import logging as _logging
        _gc_log = _logging.getLogger(__name__)

        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {"use_reentrant": False}
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

        # Verify GC is active on the inner backbone model.
        # HF sets gradient_checkpointing on Qwen3Model (inner), not the CausalLM.
        causal_lm = _get_causal_lm(self.model)
        inner = getattr(causal_lm, 'model', causal_lm)
        gc_attr = getattr(inner, 'gradient_checkpointing', None)
        gc_func = getattr(inner, '_gradient_checkpointing_func', None)
        _gc_log.info(
            "GC verify: inner.gradient_checkpointing=%s, has_gc_func=%s, "
            "inner.training=%s, inner_type=%s",
            gc_attr, gc_func is not None, inner.training, type(inner).__name__,
        )

        # Force-enable if HF's method didn't set it
        if not gc_attr and gc_func is None:
            import functools
            _gc_log.warning("GC not detected on inner model — forcing it on!")
            inner.gradient_checkpointing = True
            inner._gradient_checkpointing_func = functools.partial(
                torch_checkpoint, **gradient_checkpointing_kwargs,
            )
            _gc_log.info("GC forced: inner.gradient_checkpointing=%s",
                         inner.gradient_checkpointing)

        # GC requires model.training=True. Ensure the model is in train mode.
        if not inner.training:
            _gc_log.warning("Model in eval mode — calling train() for GC to work")
            self.model.train()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
