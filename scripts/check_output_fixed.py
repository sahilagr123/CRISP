"""Test if fixing use_cache in config.json fixes the gibberish output."""
import json
import os
import shutil

MODEL_DIR = "checkpoints/hf_weights/alice_hf"
CONFIG = os.path.join(MODEL_DIR, "config.json")

# Patch config.json: set use_cache=True
with open(CONFIG) as f:
    config = json.load(f)

print(f"use_cache before: {config.get('use_cache')}")
config["use_cache"] = True
with open(CONFIG, "w") as f:
    json.dump(config, f, indent=2)
print(f"use_cache after:  {config.get('use_cache')}")
print()

# Now test generation
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

llm = LLM(MODEL_DIR, max_model_len=6144, enforce_eager=True)
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

ids = tok.apply_chat_template(chat, add_generation_prompt=True)
out = llm.generate(
    [TokensPrompt(prompt_token_ids=ids)],
    SamplingParams(max_tokens=512, temperature=0.8),
)
print("=== OUTPUT (with use_cache=True) ===")
print(out[0].outputs[0].text[:2000])
