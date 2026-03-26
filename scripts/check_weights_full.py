"""Find the tensors with the LARGEST differences between saved and base weights."""
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
print(f"Loaded {len(trained_sd)} tensors")

print("=== Loading base weights ===")
from huggingface_hub import snapshot_download
base_path = snapshot_download(BASE, allow_patterns="*.safetensors")
base_files = sorted(glob.glob(os.path.join(base_path, "model-*.safetensors")))
base_sd = {}
for f in base_files:
    base_sd.update(load_file(f))
print(f"Loaded {len(base_sd)} tensors")

# Compare ALL tensors, rank by max absolute difference
print("\n=== All tensors ranked by max difference ===")
diffs = []
for key in sorted(base_sd.keys()):
    if key not in trained_sd:
        print(f"  MISSING: {key}")
        continue
    t = trained_sd[key].float()
    b = base_sd[key].float()
    if t.shape != b.shape:
        print(f"  SHAPE MISMATCH: {key} trained={list(t.shape)} base={list(b.shape)}")
        continue
    d = (t - b).abs()
    diffs.append((d.max().item(), d.mean().item(), key, list(t.shape)))

# Sort by max diff descending
diffs.sort(key=lambda x: x[0], reverse=True)

print(f"\nTop 20 largest differences:")
for max_d, mean_d, name, shape in diffs[:20]:
    print(f"  max={max_d:.8f}  mean={mean_d:.8f}  {name}  {shape}")

print(f"\nBottom 5 (smallest diffs):")
for max_d, mean_d, name, shape in diffs[-5:]:
    print(f"  max={max_d:.8f}  mean={mean_d:.8f}  {name}  {shape}")

# Check for any identical tensors
identical = [(name, shape) for max_d, mean_d, name, shape in diffs if max_d == 0.0]
print(f"\nIdentical tensors: {len(identical)}/{len(diffs)}")

# Stats
max_diffs = [d[0] for d in diffs]
print(f"\nOverall max diff across all tensors: {max(max_diffs):.8f}")
print(f"Overall min max-diff: {min(max_diffs):.8f}")
unique_maxes = sorted(set(f"{d:.8f}" for d in max_diffs))
print(f"Unique max-diff values: {unique_maxes[:10]}")
