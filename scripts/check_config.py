"""Compare saved model config.json against base model to find differences."""
import json

from huggingface_hub import hf_hub_download

SAVED = "checkpoints/hf_weights/alice_hf/config.json"
BASE_REPO = "Qwen/Qwen3-4B-Instruct-2507"

with open(SAVED) as f:
    saved = json.load(f)

base_path = hf_hub_download(BASE_REPO, "config.json")
with open(base_path) as f:
    base = json.load(f)

all_keys = sorted(set(list(saved.keys()) + list(base.keys())))
diffs = []
for k in all_keys:
    sv = saved.get(k)
    bv = base.get(k)
    if sv != bv:
        diffs.append((k, sv, bv))

if diffs:
    print("=== Config differences (saved vs base) ===")
    for k, sv, bv in diffs:
        print(f"  {k}:")
        print(f"    saved: {sv}")
        print(f"    base:  {bv}")
else:
    print("Configs are identical")
