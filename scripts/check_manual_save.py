"""Test: manually rebuild safetensors from trained weights using base model's structure.

Instead of save_pretrained (which produces broken safetensors), load the
trained state_dict and save it using the base model's exact shard layout.
"""
import glob
import os
import shutil
import tempfile

import torch
from safetensors.torch import load_file, save_file

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

# Load trained weights (raw tensors)
print("Loading trained weights...")
trained_sd = {}
for f in sorted(glob.glob(os.path.join(SAVED, "model-*.safetensors"))):
    trained_sd.update(load_file(f))
print(f"  {len(trained_sd)} tensors loaded")

# Create output directory: copy ALL base files
out_dir = tempfile.mkdtemp(prefix="crisp_manual_")
print(f"\nCreating manual save dir: {out_dir}")

for f in os.listdir(base_path):
    src = os.path.join(base_path, f)
    dst = os.path.join(out_dir, f)
    if os.path.isfile(src):
        shutil.copy2(src, dst)

# Now rebuild each shard: load base shard, replace with trained values, save
import json
with open(os.path.join(base_path, "model.safetensors.index.json")) as f:
    index = json.load(f)

# Group parameters by shard (from base index)
shard_to_params = {}
for param_name, shard_file in index["weight_map"].items():
    shard_to_params.setdefault(shard_file, []).append(param_name)

for shard_file, param_names in sorted(shard_to_params.items()):
    shard_tensors = {}
    for name in param_names:
        if name in trained_sd:
            shard_tensors[name] = trained_sd[name]
        else:
            print(f"  WARNING: {name} not in trained weights, using base")
            base_shard = load_file(os.path.join(base_path, shard_file))
            shard_tensors[name] = base_shard[name]

    out_path = os.path.join(out_dir, shard_file)
    save_file(shard_tensors, out_path)
    print(f"  Rebuilt {shard_file}: {len(shard_tensors)} tensors")

# Test
print(f"\n{'='*60}")
print("Testing MANUAL SAVE: trained weights in base shard structure")
print(f"{'='*60}")

llm = LLM(out_dir, max_model_len=6144, enforce_eager=True)
tok = llm.get_tokenizer()
ids = tok.apply_chat_template(PROMPT_CHAT, add_generation_prompt=True)
out = llm.generate([TokensPrompt(prompt_token_ids=ids)], PARAMS)
text = out[0].outputs[0].text

coherent = "boxed" in text or "5" in text[:200]
print(f"Coherent: {coherent}")
print(f"Output: {text[:500]}")

shutil.rmtree(out_dir)

if coherent:
    print("\n>>> FIX: use safetensors.torch.save_file() with base shard layout")
    print(">>> instead of model.save_pretrained()")
else:
    print("\n>>> Still broken — weights themselves are the issue, not file structure")
