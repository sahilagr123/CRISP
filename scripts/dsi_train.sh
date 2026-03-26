#!/bin/bash
#SBATCH --job-name=crisp
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=crisp_%j.log
#SBATCH --error=crisp_%j.log
#SBATCH --signal=B:TERM@600
#SBATCH --exclude=q001
#SBATCH --requeue

set -euo pipefail

# Forward SIGTERM to the Python child so the SIGTERM handler can save weights.
# Slurm's --signal=B:TERM@600 sends to the batch shell; this trap ensures
# the signal reaches the training process.
CHILD_PID=
trap 'echo "SIGTERM received by shell, forwarding to PID $CHILD_PID"; kill -TERM "$CHILD_PID" 2>/dev/null; wait "$CHILD_PID"; echo "Requeuing job for auto-resume..."; scontrol requeue "$SLURM_JOB_ID" 2>/dev/null || true; exit 0' TERM

echo "=== CRISP Training on DSI Cluster (H200) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# --- Setup ---
cd /home/abaumgartner/CRISP

# CUDA 13.1 on DSI is forward-compatible with torch's CUDA 12.8 —
# skip DeepSpeed's overly strict version check for JIT compilation.
export DS_SKIP_CUDA_CHECK=1

# Redirect caches off home dir to avoid quota issues.
# Node-local /tmp (~98GB) is faster than NFS home.
# HF models (~36GB) + Ray object store + weight sync temps all go here.
export RAY_TMPDIR=/tmp/crisp_ray
export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp/crisp_tmp
# Clean stale Ray/temp state from previous jobs on this node
# Clean ALL stale temp state (from any previous CRISP job on this node)
rm -rf /tmp/ray /tmp/crisp_ray /tmp/crisp_tmp /tmp/crisp_hf_weights /tmp/crisp_hf_weights_base /tmp/hf_cache 2>/dev/null || true
mkdir -p "$RAY_TMPDIR" "$HF_HOME" "$TMPDIR"

# Create venv on first run, reuse after
VENV=/home/abaumgartner/crisp_env
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# Install deps (fast if already installed)
pip install -q -e ".[infra,eval]"

# --- Clean up corrupted checkpoints on home dir ---
if [ -d "checkpoints/ds/alice" ]; then
    echo "WARNING: Removing stale DS checkpoint from home dir (corrupted by quota kill)"
    rm -rf checkpoints/ds
fi

# --- Run ---
# Strategy:
#   - --save-hf /tmp/crisp_hf_weights  → final full save (all 3 models, ~44GB)
#   - save_freq=10 (from yaml) triggers player-only HF saves (~16GB) to HOME dir
#   - SIGTERM handler also saves players to HOME dir
#   - After training, copy final weights from /tmp to home
#
# Home dir usage: ~16GB (player weights, overwritten in place each save)
# /tmp usage: ~44GB (full save at end only)
HF_TMP=/tmp/crisp_hf_weights
HF_HOME_DIR=checkpoints/hf_weights
mkdir -p "$HF_HOME_DIR"

# Auto-resume: if a previous run saved weights, resume from there
RESUME_ARGS=""
if [ -f "$HF_HOME_DIR/iteration.txt" ]; then
    PREV_ITER=$(cat "$HF_HOME_DIR/iteration.txt")
    echo "Found checkpoint at iteration $PREV_ITER — resuming"
    RESUME_ARGS="--resume-hf $HF_HOME_DIR"
else
    echo "No checkpoint found — starting fresh"
fi

echo ""
echo "Periodic player saves: $HF_HOME_DIR (every 10 iters, ~16GB)"
echo "Final full save: $HF_TMP (copied to home after training)"
echo "Starting training at $(date)..."

python -m crisp.train \
    --config configs/dsi_h200.yaml \
    --save-hf "$HF_TMP" \
    --save-hf-home "$HF_HOME_DIR" \
    --override training.num_iterations=60 \
    $RESUME_ARGS &
CHILD_PID=$!
wait "$CHILD_PID"

echo "Training finished at $(date)"

# Copy final HF weights from /tmp to home dir (players only — coach too large)
echo "Copying player HF weights to home dir..."
cp -r "$HF_TMP/alice_hf" "$HF_HOME_DIR/" 2>/dev/null && echo "  alice_hf copied" || echo "  alice_hf not found"
cp -r "$HF_TMP/bob_hf" "$HF_HOME_DIR/" 2>/dev/null && echo "  bob_hf copied" || echo "  bob_hf not found"
cp "$HF_TMP/iteration.txt" "$HF_HOME_DIR/" 2>/dev/null || true
echo "Done. Player weights in $HF_HOME_DIR/"
du -sh "$HF_HOME_DIR/"*/ 2>/dev/null || true
