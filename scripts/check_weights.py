"""Diagnose saved weights: compare against base model for corruption."""
import torch
from safetensors.torch import load_file
import os
import glob

TRAINED = "checkpoints/hf_weights/alice_hf"
BASE = "Qwen/Qwen3-4B-Instruct-2507"

print("=== Loading trained weights ===")
trained_files = sorted(glob.glob(os.path.join(TRAINED, "model-*.safetensors")))
trained_sd = {}
for f in trained_files:
    trained_sd.update(load_file(f))
print(f"Loaded {len(trained_sd)} tensors from {len(trained_files)} shards")

print("\n=== Loading base weights ===")
from huggingface_hub import snapshot_download
base_path = snapshot_download(BASE, allow_patterns="*.safetensors")
base_files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
base_sd = {}
for f in base_files:
    base_sd.update(load_file(f))
print(f"Loaded {len(base_sd)} tensors from {len(base_files)} shards")

# Check for NaN/Inf
print("\n=== NaN/Inf check ===")
nan_count = 0
inf_count = 0
for name, tensor in trained_sd.items():
    if torch.isnan(tensor).any():
        print(f"  NaN in {name}: {torch.isnan(tensor).sum().item()} values")
        nan_count += 1
    if torch.isinf(tensor).any():
        print(f"  Inf in {name}: {torch.isinf(tensor).sum().item()} values")
        inf_count += 1
if nan_count == 0 and inf_count == 0:
    print("  No NaN or Inf found")

# Compare key layers
print("\n=== Parameter comparison (trained vs base) ===")
keys_to_check = [
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.15.self_attn.q_proj.weight",
    "model.layers.35.self_attn.q_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
]

for key in keys_to_check:
    if key not in trained_sd:
        print(f"  {key}: MISSING from trained")
        continue
    if key not in base_sd:
        print(f"  {key}: MISSING from base")
        continue
    t = trained_sd[key].float()
    b = base_sd[key].float()
    diff = (t - b).abs()
    cosine = torch.nn.functional.cosine_similarity(
        t.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()
    print(f"  {key}:")
    print(f"    shape={list(t.shape)}, dtype={trained_sd[key].dtype}")
    print(f"    trained: mean={t.mean():.6f}, std={t.std():.6f}, min={t.min():.6f}, max={t.max():.6f}")
    print(f"    base:    mean={b.mean():.6f}, std={b.std():.6f}, min={b.min():.6f}, max={b.max():.6f}")
    print(f"    diff:    mean={diff.mean():.6f}, max={diff.max():.6f}, cosine_sim={cosine:.6f}")
    print(f"    identical={torch.equal(t, b)}")

# Overall stats
print("\n=== Overall ===")
identical = 0
different = 0
missing = 0
for key in base_sd:
    if key not in trained_sd:
        missing += 1
    elif torch.equal(trained_sd[key], base_sd[key]):
        identical += 1
    else:
        different += 1
print(f"  Identical: {identical}/{len(base_sd)}")
print(f"  Different: {different}/{len(base_sd)}")
print(f"  Missing:   {missing}/{len(base_sd)}")
