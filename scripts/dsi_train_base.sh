#!/bin/bash
#SBATCH --job-name=crisp-base
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=crisp_base_%j.log
#SBATCH --error=crisp_base_%j.log
#SBATCH --signal=B:TERM@600
#SBATCH --exclude=q001
#SBATCH --requeue

set -euo pipefail

CHILD_PID=
trap 'echo "SIGTERM received by shell, forwarding to PID $CHILD_PID"; kill -TERM "$CHILD_PID" 2>/dev/null; wait "$CHILD_PID"; exit $?' TERM

echo "=== CRISP Training — BASE MODEL — on DSI Cluster (H200) ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo ""

# --- Setup ---
cd /home/abaumgartner/CRISP
export DS_SKIP_CUDA_CHECK=1

export RAY_TMPDIR=/tmp/crisp_ray
export HF_HOME=/tmp/hf_cache
export TMPDIR=/tmp/crisp_tmp
# Clean ALL stale temp state (from any previous CRISP job on this node)
rm -rf /tmp/ray /tmp/crisp_ray /tmp/crisp_tmp /tmp/crisp_hf_weights /tmp/crisp_hf_weights_base /tmp/hf_cache 2>/dev/null || true
mkdir -p "$RAY_TMPDIR" "$HF_HOME" "$TMPDIR"

VENV=/home/abaumgartner/crisp_env
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv..."
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q -e ".[infra,eval]"

# --- Run ---
HF_TMP=/tmp/crisp_hf_weights_base
HF_HOME_DIR=checkpoints/hf_weights_base
mkdir -p "$HF_HOME_DIR"

# Auto-resume
RESUME_ARGS=""
if [ -f "$HF_HOME_DIR/iteration.txt" ]; then
    PREV_ITER=$(cat "$HF_HOME_DIR/iteration.txt")
    echo "Found checkpoint at iteration $PREV_ITER — resuming"
    RESUME_ARGS="--resume-hf $HF_HOME_DIR"
else
    echo "No checkpoint found — starting fresh"
fi

echo ""
echo "Config: configs/dsi_h200_base.yaml (Qwen3-4B BASE)"
echo "Periodic player saves: $HF_HOME_DIR (every 10 iters, ~16GB)"
echo "Final full save: $HF_TMP (copied to home after training)"
echo "Starting training at $(date)..."

python -m crisp.train \
    --config configs/dsi_h200_base.yaml \
    --save-hf "$HF_TMP" \
    --save-hf-home "$HF_HOME_DIR" \
    $RESUME_ARGS &
CHILD_PID=$!
wait "$CHILD_PID"

echo "Training finished at $(date)"

echo "Copying player HF weights to home dir..."
cp -r "$HF_TMP/alice_hf" "$HF_HOME_DIR/" 2>/dev/null && echo "  alice_hf copied" || echo "  alice_hf not found"
cp -r "$HF_TMP/bob_hf" "$HF_HOME_DIR/" 2>/dev/null && echo "  bob_hf copied" || echo "  bob_hf not found"
cp "$HF_TMP/iteration.txt" "$HF_HOME_DIR/" 2>/dev/null || true
echo "Done. Player weights in $HF_HOME_DIR/"
du -sh "$HF_HOME_DIR/"*/ 2>/dev/null || true
