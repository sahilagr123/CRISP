"""GRPO loss with DCPO token-adaptive clipping and JS-divergence."""
from __future__ import annotations

import torch
from torch import Tensor


def compute_grpo_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    ref_log_probs: Tensor,
    advantages: Tensor,
    attention_mask: Tensor,
    dcpo_alpha: float = 3.0,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    js_beta: float = 0.001,
) -> Tensor:
    """Compute GRPO loss with asymmetric DCPO clipping and optional JS-divergence.

    Args:
        current_log_probs: [B, T] log pi_theta(a_t | s_t) -- current policy
        old_log_probs: [B, T] log pi_old(a_t | s_t) -- rollout policy
        ref_log_probs: [B, T] log pi_ref(a_t | s_t) -- frozen reference
        advantages: [B] per-sequence advantages
        attention_mask: [B, T] valid token mask
        dcpo_alpha: DCPO sensitivity (higher -> wider bounds for rare tokens)
        clip_low: Lower clip epsilon (standard PPO/GRPO bound)
        clip_high: Upper clip epsilon (DAPO clip-higher: prevents entropy collapse)
        js_beta: JS-divergence coefficient (higher for coach due to self-referencing)

    Returns:
        Scalar loss value (token-level mean over valid tokens).
    """
    # Importance ratio
    rho = (current_log_probs - old_log_probs).exp()  # [B, T]

    # DCPO: token-adaptive clip bounds from REFERENCE policy
    prior_prob = ref_log_probs.exp().clamp(min=1e-10, max=1.0)  # [B, T]
    dcpo_scale = 1.0 + dcpo_alpha * (1.0 - prior_prob)  # [B, T]
    eps_lo = clip_low * dcpo_scale   # [B, T]
    eps_hi = clip_high * dcpo_scale  # [B, T] — wider upper bound

    # Expand per-sequence advantages to token level
    adv = advantages.unsqueeze(1).expand_as(rho)  # [B, T]

    # Clipped surrogate loss with asymmetric bounds
    surr1 = rho * adv
    surr2 = rho.clamp(1.0 - eps_lo, 1.0 + eps_hi) * adv
    policy_loss = -torch.min(surr1, surr2)  # [B, T]

    # JS-divergence (single-sample estimator)
    if js_beta > 0.0:
        p = current_log_probs.exp().clamp(min=1e-10)
        q = ref_log_probs.exp().clamp(min=1e-10)
        m = 0.5 * (p + q) + 1e-8
        log_m = m.log()
        js_div = (0.5 * (p * (p.log() - log_m) + q * (q.log() - log_m))).clamp(min=0.0)
    else:
        js_div = torch.zeros_like(policy_loss)

    # Token-level mean over valid tokens
    total_loss = (policy_loss + js_beta * js_div) * attention_mask
    return total_loss.sum() / (attention_mask.sum() + 1e-10)
