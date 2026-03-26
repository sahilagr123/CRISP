"""Check safetensors index files for weight tying issues (lm_head vs embed_tokens)."""
import json
import os

from huggingface_hub import hf_hub_download

SAVED = "checkpoints/hf_weights/alice_hf"
BASE_REPO = "Qwen/Qwen3-4B-Instruct-2507"

# Saved model index
saved_index_path = os.path.join(SAVED, "model.safetensors.index.json")
with open(saved_index_path) as f:
    saved_idx = json.load(f)
saved_wm = saved_idx.get("weight_map", {})

# Base model index
base_index_path = hf_hub_download(BASE_REPO, "model.safetensors.index.json")
with open(base_index_path) as f:
    base_idx = json.load(f)
base_wm = base_idx.get("weight_map", {})

print("=== SAVED model index ===")
print(f"Total params: {len(saved_wm)}")
print(f"lm_head.weight -> {saved_wm.get('lm_head.weight', 'MISSING')}")
print(f"model.embed_tokens.weight -> {saved_wm.get('model.embed_tokens.weight', 'MISSING')}")
lm_keys = [k for k in saved_wm if "lm_head" in k]
print(f"All lm_head keys: {lm_keys if lm_keys else 'NONE'}")

print(f"\n=== BASE model index ===")
print(f"Total params: {len(base_wm)}")
print(f"lm_head.weight -> {base_wm.get('lm_head.weight', 'MISSING')}")
print(f"model.embed_tokens.weight -> {base_wm.get('model.embed_tokens.weight', 'MISSING')}")
lm_keys_base = [k for k in base_wm if "lm_head" in k]
print(f"All lm_head keys: {lm_keys_base if lm_keys_base else 'NONE'}")

# Diff: keys in one but not the other
saved_only = set(saved_wm.keys()) - set(base_wm.keys())
base_only = set(base_wm.keys()) - set(saved_wm.keys())
print(f"\n=== Index differences ===")
print(f"Keys only in saved ({len(saved_only)}): {sorted(saved_only) if saved_only else 'none'}")
print(f"Keys only in base ({len(base_only)}):  {sorted(base_only) if base_only else 'none'}")
