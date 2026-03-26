"""Isolate whether the issue is the tokenizer or the weights.

Test 1: Saved weights + base tokenizer
Test 2: Base weights + saved tokenizer
"""
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import gc
import torch

PROMPT_CHAT = [
    {"role": "system", "content": (
        "You are a methodical math solver. Work through problems step by step. "
        "Show your work, then put your final numerical answer within \\boxed{}. "
        "You MUST end your response with \\boxed{answer}."
    )},
    {"role": "user", "content": (
        "What is 2 + 3?\n\n"
        "Solve this problem. Show your reasoning, then give your final answer "
        "as \\boxed{answer}."
    )},
]

PARAMS = SamplingParams(max_tokens=256, temperature=0.0)


def test(model_path, tokenizer_path, label):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"  model:     {model_path}")
    print(f"  tokenizer: {tokenizer_path}")
    print(f"{'='*60}")

    llm = LLM(
        model_path,
        tokenizer=tokenizer_path,
        max_model_len=6144,
        enforce_eager=True,
    )
    tok = llm.get_tokenizer()
    ids = tok.apply_chat_template(PROMPT_CHAT, add_generation_prompt=True)
    print(f"Prompt tokens: {len(ids)}")

    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], PARAMS)
    text = out[0].outputs[0].text
    coherent = "boxed" in text or "5" in text[:200]

    print(f"Coherent: {coherent}")
    print(f"Output: {text[:500]}")

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return coherent


SAVED = "checkpoints/hf_weights/alice_hf"
BASE = "Qwen/Qwen3-4B-Instruct-2507"

r1 = test(SAVED, BASE, "TEST 1: Saved weights + BASE tokenizer")
r2 = test(BASE, SAVED, "TEST 2: Base weights + SAVED tokenizer")

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Saved weights + base tokenizer:  {'OK' if r1 else 'BROKEN'}")
print(f"Base weights + saved tokenizer:  {'OK' if r2 else 'BROKEN'}")
if r1 and not r2:
    print("\n>>> TOKENIZER is the problem")
elif not r1 and r2:
    print("\n>>> WEIGHTS are the problem")
elif r1 and r2:
    print("\n>>> Both work in isolation — interaction issue")
else:
    print("\n>>> Both broken — deeper issue")
