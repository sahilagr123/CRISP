"""Modal wrapper for CRISP smoke test.

Usage:
    modal run scripts/modal_smoke.py          # run detached, check volume later
    modal volume get crisp-smoke-reports smoke_report.md .  # fetch report
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
    # Pre-download models into image layer (avoids download per run)
    .run_commands(
        f"huggingface-cli download {PLAYER_MODEL}",
        f"huggingface-cli download {COACH_MODEL}",
    )
    .add_local_dir("crisp", "/root/project/crisp")
    .add_local_dir("configs", "/root/project/configs")
    .add_local_dir("scripts", "/root/project/scripts")
    .add_local_file("pyproject.toml", "/root/project/pyproject.toml")
)

app = modal.App("crisp-smoke-test", image=image)
vol = modal.Volume.from_name("crisp-smoke-reports", create_if_missing=True)


@app.function(
    gpu="H100:4",
    timeout=18000,  # 5 hours
    volumes={"/reports": vol},
)
def run_smoke_test():
    import os
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")

    from scripts.smoke_test import main

    output_path = "/reports/smoke_report.md"
    main(["--config", "configs/smoke_test.yaml", "--output", output_path])

    # Commit volume so report persists
    vol.commit()

    with open(output_path) as f:
        report = f.read()

    print(report)
    return report


@app.local_entrypoint()
def main():
    print("Running smoke test on 4×H100...")
    print("(Init ~5min, ~5min/iter)")
    print()

    report = run_smoke_test.remote()

    with open("smoke_report.md", "w") as f:
        f.write(report)
    print(f"\nReport saved to smoke_report.md ({len(report)} chars)")
