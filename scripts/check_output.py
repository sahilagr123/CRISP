"""Quick diagnostic: see what the trained model actually generates."""
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM("checkpoints/hf_weights/alice_hf", max_model_len=6144, enforce_eager=True)
tok = llm.get_tokenizer()

chat = [
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

# With enable_thinking=False (matches training)
try:
    ids_no_think = tok.apply_chat_template(chat, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    ids_no_think = tok.apply_chat_template(chat, add_generation_prompt=True)

# Without enable_thinking (Qwen3 default = thinking ON)
ids_default = tok.apply_chat_template(chat, add_generation_prompt=True)

print("=== PROMPT COMPARISON ===")
print(f"enable_thinking=False tokens: {len(ids_no_think)}")
print(f"default tokens:               {len(ids_default)}")
print(f"Same? {ids_no_think == ids_default}")
print()
print("Last 30 tokens (no_think):", repr(tok.decode(ids_no_think[-30:])))
print("Last 30 tokens (default): ", repr(tok.decode(ids_default[-30:])))
print()

# Generate with both prompts
params = SamplingParams(max_tokens=512, temperature=0.8)

print("=== OUTPUT (enable_thinking=False) ===")
out1 = llm.generate([TokensPrompt(prompt_token_ids=ids_no_think)], params)
print(out1[0].outputs[0].text[:2000])
print()

print("=== OUTPUT (default / thinking ON) ===")
out2 = llm.generate([TokensPrompt(prompt_token_ids=ids_default)], params)
print(out2[0].outputs[0].text[:2000])
