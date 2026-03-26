"""Extract HF-format weights from DeepSpeed ZeRO-2 checkpoints.

Usage:
    python scripts/extract_hf_from_ds.py checkpoints/ds checkpoints/hf_weights

This loads the model_states from each DS checkpoint (player/coach),
strips the Actor wrapper prefix, and saves as HF-compatible safetensors
with the matching tokenizer. Much smaller than the full DS checkpoint
(no optimizer state).

Memory-efficient: copies config/tokenizer from the base model without
loading the full base model weights into RAM.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoConfig


PLAYER_BASE = "Qwen/Qwen3-4B-Instruct-2507"
COACH_BASE = "Qwen/Qwen3-14B"


def extract_model(ds_ckpt_dir: str, output_dir: str, base_model: str, tag: str = "ckpt") -> int:
    """Extract HF weights from a DS checkpoint directory.

    Memory-efficient: never loads the base model weights. Instead:
    1. Load DS checkpoint model_states (~28GB for 14B)
    2. Strip Actor prefix from keys
    3. Save directly as safetensors shards
    4. Copy config/tokenizer from base model

    Returns the iteration number from the checkpoint.
    """
    model_states_path = os.path.join(ds_ckpt_dir, tag, "mp_rank_00_model_states.pt")
    if not os.path.isfile(model_states_path):
        print(f"ERROR: {model_states_path} not found")
        sys.exit(1)

    print(f"Loading {model_states_path} ...")
    sd = torch.load(model_states_path, map_location="cpu", weights_only=False)
    iteration = sd.get("iteration", 0)
    print(f"  Checkpoint iteration: {iteration}")

    # Strip the Actor wrapper prefix: "model.xxx" -> "xxx"
    raw_state = sd["module"]
    del sd  # free everything except module
    gc.collect()

    hf_state = {}
    keys_to_delete = []
    for key in list(raw_state.keys()):
        value = raw_state[key]
        if key.startswith("model."):
            hf_key = key[len("model."):]
        else:
            hf_key = key
        hf_state[hf_key] = value
        del raw_state[key]  # free as we go

    del raw_state
    gc.collect()

    print(f"  {len(hf_state)} parameters")

    # Save as safetensors
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Saving to {output_dir} ...")
    save_file(hf_state, os.path.join(output_dir, "model.safetensors"))

    # Write safetensors index for compatibility
    total_size = sum(v.numel() * v.element_size() for v in hf_state.values())
    weight_map = {k: "model.safetensors" for k in hf_state.keys()}
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    del hf_state
    gc.collect()

    # Copy config from base model (tiny download, no weights)
    print(f"  Copying config and tokenizer from {base_model} ...")
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    config.save_pretrained(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)

    print(f"  Done.")
    return iteration


def main():
    parser = argparse.ArgumentParser(description="Extract HF weights from DS checkpoints")
    parser.add_argument("ds_dir", help="DS checkpoint root (contains player/ and coach/ subdirs)")
    parser.add_argument("output_dir", help="Output directory for HF weights")
    parser.add_argument("--tag", default="ckpt", help="DS checkpoint tag (default: ckpt)")
    parser.add_argument("--player-only", action="store_true", help="Only extract player")
    parser.add_argument("--coach-only", action="store_true", help="Only extract coach")
    args = parser.parse_args()

    # Support both new (alice/bob) and legacy (player) checkpoint layouts
    alice_ds = os.path.join(args.ds_dir, "alice")
    bob_ds = os.path.join(args.ds_dir, "bob")
    legacy_player_ds = os.path.join(args.ds_dir, "player")
    coach_ds = os.path.join(args.ds_dir, "coach")

    alice_out = os.path.join(args.output_dir, "alice_hf")
    bob_out = os.path.join(args.output_dir, "bob_hf")
    coach_out = os.path.join(args.output_dir, "coach_hf")

    iteration = 0

    if not args.coach_only:
        if os.path.isdir(alice_ds):
            print("=== Extracting Alice weights ===")
            iteration = extract_model(alice_ds, alice_out, PLAYER_BASE, tag=args.tag)
            gc.collect()
            print()

            if os.path.isdir(bob_ds):
                print("=== Extracting Bob weights ===")
                extract_model(bob_ds, bob_out, PLAYER_BASE, tag=args.tag)
                gc.collect()
                print()
        elif os.path.isdir(legacy_player_ds):
            print("=== Extracting player weights (legacy) ===")
            player_out = os.path.join(args.output_dir, "player_hf")
            iteration = extract_model(legacy_player_ds, player_out, PLAYER_BASE, tag=args.tag)
            gc.collect()
            print()

    if not args.player_only:
        print("=== Extracting coach weights ===")
        coach_iter = extract_model(coach_ds, coach_out, COACH_BASE, tag=args.tag)
        if args.coach_only:
            iteration = coach_iter
        print()

    # Write iteration marker
    iter_path = os.path.join(args.output_dir, "iteration.txt")
    with open(iter_path, "w") as f:
        f.write(str(iteration))
    print(f"Iteration {iteration} written to {iter_path}")
    print(f"\nTotal output size:")
    os.system(f"du -sh {args.output_dir}/*/")


if __name__ == "__main__":
    main()
