"""Step 1: Coach problem generation and parsing.

Two-step process:
1. Generate: Coach outputs <question>...</question> (no answer)
2. Resolve: Coach solves its own problem with thinking mode to get ground truth
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import numpy as np

from crisp.infra.experience import generate_samples
from crisp.types import Problem, TokenSequence
from crisp.verifier.answer_extraction import extract_answer

logger = logging.getLogger(__name__)

# Lazy-loaded sentence-transformer model for embeddings.
_embedder = None

_QUESTION_RE = re.compile(r"<question>\s*(.*?)\s*</question>", re.DOTALL)
# Fallback: opening tag without closing (truncated output)
_QUESTION_OPEN_RE = re.compile(r"<question>\s*(.*)", re.DOTALL)
# Template placeholders that indicate the model regurgitated the prompt
_TEMPLATE_PLACEHOLDERS = {"[PROBLEM]", "(your problem here)"}
# Common preamble patterns the coach prepends when it drops <question> tags
_PREAMBLE_RE = re.compile(
    r"^(?:(?:Sure|Here|Okay|Of course)[,!.]?\s*)?(?:Here(?:'s| is) (?:a|the|one|your) "
    r"(?:math )?problem[^:]*:\s*)?",
    re.IGNORECASE,
)
# Think-block stripping (safety net — thinking is disabled for generation)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _validate_content(content: str) -> Optional[str]:
    """Validate extracted content: length >= 20, not a template placeholder."""
    content = content.strip()
    if not content or len(content) < 20:
        return None
    if content in _TEMPLATE_PLACEHOLDERS:
        return None
    return content


def _extract_question(text: str) -> Optional[str]:
    """Extract problem text from coach output.

    Three-tier extraction:
    1. Full <question>...</question> tags (normal case)
    2. Opening <question> tag only (truncated output)
    3. Tagless fallback: strip preamble, use remaining text as problem

    Tier 3 prevents coach format drift from causing a death spiral where
    valid math problems are rejected just because <question> tags are missing.
    """
    # Tier 1: Full tags
    m = _QUESTION_RE.search(text)
    if m:
        return _validate_content(m.group(1))

    # Tier 2: Truncated (opening tag, no closing)
    m = _QUESTION_OPEN_RE.search(text)
    if m:
        result = _validate_content(m.group(1))
        if result:
            return result

    # Tier 3: No tags at all — strip preamble, truncate to question only.
    # The coach's 2048-token budget can produce ~10K chars. Without tags,
    # long output is almost certainly "question + solution" — we only want
    # the question part.  A reasonable math problem statement is < 2000 chars.
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _PREAMBLE_RE.sub("", cleaned).strip()

    # Try to find a natural question boundary: double newline often separates
    # the problem statement from the coach's solution/explanation.
    _MAX_QUESTION_LEN = 2000
    if len(cleaned) > _MAX_QUESTION_LEN:
        # Look for a double-newline break within the first 2000 chars
        break_pos = cleaned.find("\n\n", 100)  # at least 100 chars in
        if 100 < break_pos < _MAX_QUESTION_LEN:
            cleaned = cleaned[:break_pos].strip()
        else:
            cleaned = cleaned[:_MAX_QUESTION_LEN].strip()

    result = _validate_content(cleaned)
    if result:
        logger.info("Tagless fallback extracted question (len=%d)", len(result))
    return result


def _get_embedder():
    """Lazy-load the sentence-transformer model."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _embedder


def _build_solve_prompts(ctx, problem_texts: List[str]) -> List[List[int]]:
    """Build tokenized prompts for the coach to solve its own problems."""
    from crisp.workflow.tokenizer import _apply_chat

    tokenizer = getattr(ctx, 'coach_tokenizer', None) or ctx.tokenizer
    ccfg = ctx.config.coach
    prompts = []
    for text in problem_texts:
        user_msg = ccfg.coach_solve_prompt_template.format(problem=text)
        token_ids = _apply_chat(
            tokenizer, ccfg.coach_solve_system_prompt, user_msg,
            enable_thinking=True,
        )
        prompts.append(list(token_ids))
    return prompts


def generate_problems(
    ctx,
    coach_prompts: Optional[List[List[int]]] = None,
    accuracy_history: Optional[List[float]] = None,
) -> List[Problem]:
    """Generate problems from the coach model in two steps.

    Step 1 (Generate): Coach outputs <question>...</question> only.
    Step 2 (Resolve): Coach solves each problem with thinking mode
                      to produce a verified \\boxed{} ground truth.

    Skips any problem where either step fails to parse.
    """
    if coach_prompts is None:
        coach_prompts = _build_coach_prompts(ctx, accuracy_history=accuracy_history)

    coach_temp = ctx.config.coach.coach_temperature

    # --- Step 1: Generate questions ---
    if ctx.coach_vllm is not None:
        gen_rollouts = generate_samples(
            ctx.coach_vllm,
            prompt_token_ids=coach_prompts,
            problem_indices=list(range(len(coach_prompts))),
            player_id=-1,
            max_new_tokens=2048,  # Budget for preamble + question text
            temperature=coach_temp,
        )
    else:
        from crisp.infra.hf_generate import generate_from_ds_model
        coach_tok = getattr(ctx, 'coach_tokenizer', None) or ctx.tokenizer
        gen_rollouts = generate_from_ds_model(
            ctx.ds_coach, coach_tok, coach_prompts, max_new_tokens=2048,
            temperature=coach_temp,
        )

    # Parse questions from generation output
    questions: List[str] = []
    gen_sequences: List[TokenSequence] = []
    for rollout in gen_rollouts:
        text = _extract_question(rollout.text)
        if text is None or not text.strip():
            logger.warning("Coach output unparseable, skipping: %.200s",
                           rollout.text[:200])
            continue
        questions.append(text)
        gen_sequences.append(TokenSequence(
            tokens=rollout.tokens,
            log_probs=rollout.log_probs,
            text=rollout.text,
        ))

    if not questions:
        logger.warning("No valid questions generated by coach")
        return []

    logger.info("Coach generated %d questions, now resolving...", len(questions))

    # --- Step 2: Resolve (coach solves its own problems) ---
    solve_prompts = _build_solve_prompts(ctx, questions)

    # Use low (but nonzero) temperature to avoid greedy repetition loops
    # while keeping outputs reliable.
    solve_temp = 0.1

    solve_budget = ctx.config.coach.coach_solve_max_new_tokens

    if ctx.coach_vllm is not None:
        solve_rollouts = generate_samples(
            ctx.coach_vllm,
            prompt_token_ids=solve_prompts,
            problem_indices=list(range(len(solve_prompts))),
            player_id=-1,
            max_new_tokens=solve_budget,
            temperature=solve_temp,
        )
    else:
        from crisp.infra.hf_generate import generate_from_ds_model
        coach_tok = getattr(ctx, 'coach_tokenizer', None) or ctx.tokenizer
        solve_rollouts = generate_from_ds_model(
            ctx.ds_coach, coach_tok, solve_prompts, max_new_tokens=solve_budget,
            temperature=solve_temp,
        )

    # Parse ground truth from solve output
    problems = []
    valid_texts = []
    n_resolved = 0
    for i, rollout in enumerate(solve_rollouts):
        ground_truth = extract_answer(rollout.text, truncated=False)
        if ground_truth is None:
            # Coach couldn't solve its own problem — include it anyway so the
            # coach gets a negative training signal (unsolvable penalty).
            # Without this, the coach never learns "that was too hard for me".
            logger.warning("Coach failed to solve problem %d (will penalize): %.200s",
                           i, rollout.text[-200:])
            problems.append(Problem(
                text=questions[i],
                ground_truth="UNSOLVABLE",
                coach_sequence=gen_sequences[i],
                self_solvable=False,
            ))
            valid_texts.append(questions[i])
            continue

        n_resolved += 1
        problems.append(Problem(
            text=questions[i],
            ground_truth=ground_truth,
            coach_sequence=gen_sequences[i],  # Use generation sequence for training
        ))
        valid_texts.append(questions[i])

    logger.info("Coach resolved %d/%d problems", n_resolved, len(questions))

    # Compute embeddings in a single batch
    if valid_texts:
        embedder = _get_embedder()
        embeddings = embedder.encode(valid_texts)
        for i, problem in enumerate(problems):
            problem.coach_embedding = embeddings[i]

    return problems


def _build_coach_prompts(
    ctx, accuracy_history: Optional[List[float]] = None,
) -> List[List[int]]:
    """Build tokenized coach prompts from the template."""
    tokenizer = getattr(ctx, 'coach_tokenizer', None) or getattr(ctx, 'tokenizer', None)
    if tokenizer is None:
        return []
    from crisp.workflow.tokenizer import build_coach_prompts
    return build_coach_prompts(
        tokenizer, ctx.config, accuracy_history=accuracy_history,
        iteration=getattr(ctx, 'iteration', 0),
    )
