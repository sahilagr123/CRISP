"""Swap safetensors files between base and saved to isolate the issue.

Test: Copy ALL non-safetensor files from base model into the saved model
directory (config, tokenizer, index, generation_config, etc.).
Keep only the safetensors weight files from the saved model.
"""
import glob
import os
import shutil
import tempfile

from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

SAVED = "checkpoints/hf_weights/alice_hf"
BASE_REPO = "Qwen/Qwen3-4B-Instruct-2507"

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

# Download base model
print("Downloading base model files...")
base_path = snapshot_download(BASE_REPO)
print(f"Base model at: {base_path}")

# Create hybrid directory: saved safetensors + ALL base metadata
hybrid_dir = tempfile.mkdtemp(prefix="crisp_hybrid_")
print(f"\nCreating hybrid dir: {hybrid_dir}")

# Copy ALL base files first
for f in os.listdir(base_path):
    src = os.path.join(base_path, f)
    dst = os.path.join(hybrid_dir, f)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"  base -> {f}")

# Overwrite with saved model's safetensors files ONLY
for f in os.listdir(SAVED):
    if f.endswith(".safetensors"):
        src = os.path.join(SAVED, f)
        dst = os.path.join(hybrid_dir, f)
        shutil.copy2(src, dst)
        print(f"  saved -> {f} (overwrite)")

print(f"\nHybrid dir contents:")
for f in sorted(os.listdir(hybrid_dir)):
    size = os.path.getsize(os.path.join(hybrid_dir, f))
    print(f"  {f:50s} {size:>12,} bytes")

# Test the hybrid model
print(f"\n{'='*60}")
print("Testing HYBRID: saved safetensors + base everything else")
print(f"{'='*60}")

llm = LLM(hybrid_dir, max_model_len=6144, enforce_eager=True)
tok = llm.get_tokenizer()
ids = tok.apply_chat_template(PROMPT_CHAT, add_generation_prompt=True)
out = llm.generate([TokensPrompt(prompt_token_ids=ids)], PARAMS)
text = out[0].outputs[0].text

coherent = "boxed" in text or "5" in text[:200]
print(f"Coherent: {coherent}")
print(f"Output: {text[:500]}")

# Cleanup
shutil.rmtree(hybrid_dir)
