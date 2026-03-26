"""Modal production smoke test for two-player architecture on 4×H200.

Runs 5 iterations with full H200 production settings (12K tokens, 49K context,
batch_size=8) to validate the ds_alice/ds_bob independent training pipeline.

Usage:
    modal run scripts/modal_two_player_smoke.py
    modal volume get crisp-smoke-reports two_player_smoke.md .

Budget: ~$10-15 on 4×H200 (~30-45 min total)
"""
from __future__ import annotations

import modal

PLAYER_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
COACH_MODEL = "Qwen/Qwen3-14B"

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
    )
    .add_local_dir("crisp", "/root/project/crisp")
    .add_local_dir("configs", "/root/project/configs")
    .add_local_dir("scripts", "/root/project/scripts")
    .add_local_file("pyproject.toml", "/root/project/pyproject.toml")
)

app = modal.App("crisp-two-player-smoke", image=image)
vol = modal.Volume.from_name("crisp-smoke-reports", create_if_missing=True)


@app.function(
    gpu="H200:4",
    timeout=7200,  # 2 hours (should finish in ~45 min)
    volumes={"/reports": vol},
)
def run_smoke():
    import os
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")

    from scripts.smoke_test import main

    output_path = "/reports/two_player_smoke.md"
    main(["--config", "configs/h200_smoke.yaml", "--output", output_path])

    vol.commit()

    with open(output_path) as f:
        report = f.read()

    print(report)
    return report


@app.local_entrypoint()
def main():
    print("Two-player production smoke test on 4×H200 (141GB each)...")
    print("  Config: configs/h200_smoke.yaml")
    print("  5 iterations, full H200 settings (12K tokens, 49K context)")
    print("  Validating: ds_alice/ds_bob, per-player weight sync,")
    print("  independent EMA trackers, sequential rollouts")
    print()

    report = run_smoke.remote()

    with open("two_player_smoke.md", "w") as f:
        f.write(report)
    print(f"\nReport saved to two_player_smoke.md ({len(report)} chars)")
