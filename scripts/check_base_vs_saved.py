"""Compare base model output vs saved model output with identical prompts."""
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

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

PARAMS = SamplingParams(max_tokens=256, temperature=0.0)  # greedy for reproducibility


def test_model(path, label):
    print(f"\n{'='*60}")
    print(f"Testing: {label}")
    print(f"Path:    {path}")
    print(f"{'='*60}")

    llm = LLM(path, max_model_len=6144, enforce_eager=True)
    tok = llm.get_tokenizer()

    ids = tok.apply_chat_template(PROMPT_CHAT, add_generation_prompt=True)
    print(f"Prompt tokens: {len(ids)}")
    print(f"Last 10 decoded: {repr(tok.decode(ids[-10:]))}")

    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], PARAMS)
    text = out[0].outputs[0].text

    print(f"\nOutput ({len(text)} chars):")
    print(text[:1000])

    # Cleanup so next model can use the GPU
    del llm
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    return text


base_out = test_model("Qwen/Qwen3-4B-Instruct-2507", "BASE MODEL")
saved_out = test_model("checkpoints/hf_weights/alice_hf", "SAVED MODEL")

print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"Base coherent:  {'boxed' in base_out or '5' in base_out[:200]}")
print(f"Saved coherent: {'boxed' in saved_out or '5' in saved_out[:200]}")
print(f"Outputs match:  {base_out[:200] == saved_out[:200]}")
