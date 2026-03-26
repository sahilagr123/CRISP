"""Modal wrapper for CRISP production training.

Saves checkpoints to a persistent volume so training can be resumed
on Modal or downloaded and continued on another cluster.

Usage:
    modal run scripts/modal_production.py                    # fresh start
    modal run scripts/modal_production.py --resume           # resume from DS checkpoint
    modal run scripts/modal_production.py --resume-hf       # resume from HF Hub weights
    modal volume get crisp-checkpoints hf_weights/ ./models/ # download trained models
    modal volume ls crisp-checkpoints                        # list saved files
"""
from __future__ import annotations

import modal

PLAYER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
COACH_MODEL = "Qwen/Qwen3-14B"

# Fine-tuned weights on HF Hub (for --resume-hf).
# Legacy single-player checkpoint — loaded as Alice, Bob starts from base.
# train.py handles backward compat (player_hf → alice + bob from scratch).
PLAYER_FINETUNED = "swisski/crisp-player-iter15"
COACH_FINETUNED = "swisski/crisp-coach-iter15"
RESUME_ITERATION = 15

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm==0.16.0",
        "deepspeed==0.16.4",
        "ray>=2.9",
        "transformers>=4.40",
        "sentence-transformers",
        "peft>=0.7",
        "numpy",
        "sympy",
        "scipy",
        "pyyaml",
        "packaging",
        "datasets",
        "mpi4py",
        "huggingface_hub",
    )
    .run_commands(
        f"huggingface-cli download {PLAYER_MODEL}",
        f"huggingface-cli download {COACH_MODEL}",
        f"huggingface-cli download {PLAYER_FINETUNED}",
        f"huggingface-cli download {COACH_FINETUNED}",
    )
    .add_local_dir("crisp", "/root/project/crisp")
    .add_local_dir("configs", "/root/project/configs")
    .add_local_dir("scripts", "/root/project/scripts")
    .add_local_file("pyproject.toml", "/root/project/pyproject.toml")
)

app = modal.App("crisp-production", image=image)
vol = modal.Volume.from_name("crisp-checkpoints", create_if_missing=True)


@app.function(
    gpu="H100:4",
    timeout=72000,  # 20 hours
    volumes={"/checkpoints": vol},
)
def run_training(resume: bool = False, resume_hf: bool = False):
    import os
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")

    from crisp.train import main

    args = ["--config", "configs/production.yaml",
            "--save-hf", "/checkpoints/hf_weights"]

    if resume:
        args += ["--resume", "/checkpoints/ds"]
    elif resume_hf:
        # HF Hub models are cached by huggingface-cli download during image build.
        # Pass repo IDs directly — transformers will resolve from cache.
        args += [
            "--override",
            f"training.model_name={PLAYER_FINETUNED}",
            f"training.coach_model_name={COACH_FINETUNED}",
            f"training.start_iteration={RESUME_ITERATION}",
        ]

    main(args)

    # Commit volume so all writes persist
    vol.commit()


@app.local_entrypoint()
def main(resume: bool = False, resume_hf: bool = False):
    mode = "resume (DS)" if resume else "resume (HF Hub)" if resume_hf else "fresh"
    print(f"Starting CRISP production training on 4×H100...")
    print(f"  Mode: {mode}")
    if resume_hf:
        print(f"  Player: {PLAYER_FINETUNED}")
        print(f"  Coach: {COACH_FINETUNED}")
        print(f"  Starting iteration: {RESUME_ITERATION}")
    print(f"  Checkpoints: crisp-checkpoints volume")
    print(f"  Config: configs/production.yaml")
    print()
    print("To download trained models after completion:")
    print("  modal volume get crisp-checkpoints hf_weights/ ./models/")
    print()

    run_training.remote(resume=resume, resume_hf=resume_hf)
    print("Training function returned.")
