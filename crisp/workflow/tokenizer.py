"""Shared tokenizer access and prompt building for CRISP workflow steps."""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from crisp.types import Problem

_tokenizer_cache: Dict[str, Any] = {}


def get_tokenizer(model_name: str) -> Any:
    """Get or create a cached HuggingFace tokenizer."""
    if model_name not in _tokenizer_cache:
        from transformers import AutoTokenizer

        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    return _tokenizer_cache[model_name]


def _apply_chat(
    tokenizer, system_prompt: str, user_message: str,
    enable_thinking: bool = False,
) -> List[int]:
    """Build token IDs using the model's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    kwargs: dict = dict(add_generation_prompt=True, tokenize=True)
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=enable_thinking, **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


_TOPIC_POOL = [
    "algebra",
    "combinatorics",
    "number theory",
    "geometry",
    "probability",
    "sequences and series",
    "modular arithmetic",
    "inequalities",
    "polynomials",
    "trigonometry",
]

_PERF_CONTEXT_WINDOW = 5  # Show last N iterations of accuracy


def _format_performance_context(accuracy_history: List[float]) -> str:
    """Format recent accuracy history into a prompt snippet for the coach."""
    recent = accuracy_history[-_PERF_CONTEXT_WINDOW:]
    if not recent:
        return ""
    mean_acc = sum(recent) / len(recent)
    lines = ["ACCURACY FEEDBACK"]
    lines.append(f"Target: 40-60% student accuracy. Recent results:")
    for i, acc in enumerate(recent):
        iter_num = len(accuracy_history) - len(recent) + i + 1
        lines.append(f"  Iteration {iter_num}: {acc:.0%}")
    lines.append(f"  Average: {mean_acc:.0%}")
    lines.append("")

    if mean_acc < 0.05:
        lines.append(
            "CRITICAL: Students cannot solve ANY of your problems. "
            "Drop to pre-AMC level: single-step arithmetic, basic one-variable "
            "equations like '2x + 3 = 11', or simple counting. "
            "The answer must be a small integer."
        )
    elif mean_acc < 0.20:
        lines.append(
            "Too hard. Drop to early AMC 8 level: single-variable equations, "
            "basic counting, simple geometry with given formulas. "
            "Problems should be solvable in 2-4 steps."
        )
    elif mean_acc < 0.35:
        lines.append(
            "Slightly too hard. Target AMC 8 / easy AMC 10: "
            "fewer steps, more concrete numbers, standard techniques."
        )
    elif mean_acc > 0.75:
        lines.append(
            "Too easy. Increase difficulty: add a twist, "
            "combine topics, require multi-step reasoning, "
            "or target competition-level problems."
        )
    else:
        lines.append(
            "Good calibration (40-60% range). Maintain this difficulty."
        )

    return "\n".join(lines) + "\n\n"


def build_coach_prompts(
    tokenizer, config, n: Optional[int] = None,
    accuracy_history: Optional[List[float]] = None,
    iteration: int = 0,
) -> List[List[int]]:
    """Tokenize n coach prompts, each with a different topic to ensure diversity.

    During warmup (iteration < warmup_iters), uses the AMC 10-anchored system
    prompt.  After warmup, switches to a pure accuracy-feedback-driven prompt.
    """
    if n is None:
        n = config.coach.batch_size

    # Select system prompt based on training phase:
    #   Phase 1 (0..warmup):  AMC 10 anchor — safe floor
    #   Phase 2 (warmup..rampup): AMC 12 anchor — push harder
    #   Phase 3 (rampup+):   Pure accuracy-driven — no difficulty anchors
    if iteration < config.coach.warmup_iters:
        system_prompt = config.coach.coach_system_prompt
    elif iteration < config.coach.rampup_iters:
        system_prompt = config.coach.coach_rampup_system_prompt
    else:
        system_prompt = config.coach.coach_post_warmup_system_prompt

    perf_prefix = ""
    if accuracy_history:
        perf_prefix = _format_performance_context(accuracy_history)

    # Shuffle topics and cycle through them so each prompt is distinct
    topics = list(_TOPIC_POOL)
    random.shuffle(topics)

    prompts = []
    for i in range(n):
        topic = topics[i % len(topics)]
        user_message = perf_prefix + config.coach.coach_prompt_template.format(
            topic=topic,
        )
        # Disable thinking for question generation — coach just needs to output
        # <question> tags.  Thinking is enabled in Step 2 (solve/resolve).
        token_ids = _apply_chat(
            tokenizer,
            system_prompt,
            user_message,
            enable_thinking=False,
        )
        prompts.append(list(token_ids))
    return prompts


def build_player_prompts(
    tokenizer,
    config,
    problems: List[Problem],
    player_id: int,
) -> List[List[int]]:
    """Tokenize problem texts with per-player system prompts and chat template.

    Returns rollouts_per_problem copies per problem.
    """
    pcfg = config.player
    rpg = pcfg.rollouts_per_problem

    if player_id == 0:
        system_prompt = pcfg.alice_system_prompt
    else:
        system_prompt = pcfg.bob_system_prompt

    prompts = []
    for problem in problems:
        user_message = pcfg.solve_prompt_template.format(problem=problem.text)
        token_ids = _apply_chat(tokenizer, system_prompt, user_message)
        for _ in range(rpg):
            prompts.append(list(token_ids))
    return prompts


def build_discussion_prompts(
    tokenizer, config, prompt_texts: List[str]
) -> List[List[int]]:
    """Tokenize discussion prompts using chat template."""
    return [
        _apply_chat(
            tokenizer,
            config.coach.discussion_system_prompt,
            text,
        )
        for text in prompt_texts
    ]
