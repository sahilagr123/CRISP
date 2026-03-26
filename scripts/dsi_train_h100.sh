#!/bin/bash
#SBATCH --job-name=crisp
#SBATCH --partition=general
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=crisp_%j.log
#SBATCH --error=crisp_%j.log

set -euo pipefail

echo "=== CRISP Training on DSI Cluster (H100) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# --- Setup ---
cd /home/abaumgartner/CRISP

export DS_SKIP_CUDA_CHECK=1

# Redirect caches off home dir to avoid quota issues.
export RAY_TMPDIR=/tmp/crisp_ray
export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp/crisp_tmp
mkdir -p "$RAY_TMPDIR" "$HF_HOME" "$TMPDIR"

# DS checkpoints to /tmp (too large for home quota)
DS_CKPT=/tmp/crisp_checkpoints
mkdir -p "$DS_CKPT"

# Create venv on first run, reuse after
VENV=/home/abaumgartner/crisp_env
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install deps (fast if already installed)
pip install -q -e ".[infra,eval]"

# Clean up corrupted checkpoints on home dir
if [ -d "checkpoints/ds/alice" ]; then
    echo "WARNING: Removing stale DS checkpoint from home dir"
    rm -rf checkpoints/ds
fi

# --- Run ---
echo ""
echo "DS checkpoints: $DS_CKPT"
echo "HF weights: checkpoints/hf_weights"
echo "Starting training at $(date)..."

if [ -d "$DS_CKPT/alice" ]; then
    echo "Resuming from /tmp checkpoint..."
    python -m crisp.train \
        --config configs/dsi_h100.yaml \
        --save-hf checkpoints/hf_weights \
        --override training.checkpoint_dir="$DS_CKPT" \
        --resume "$DS_CKPT"
else
    echo "Starting fresh training..."
    python -m crisp.train \
        --config configs/dsi_h100.yaml \
        --save-hf checkpoints/hf_weights \
        --override training.checkpoint_dir="$DS_CKPT"
fi

echo "Training finished at $(date)"
