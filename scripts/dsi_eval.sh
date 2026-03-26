#!/bin/bash
#SBATCH --job-name=crisp-eval
#SBATCH --partition=general
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=eval_%j.log
#SBATCH --error=eval_%j.log

set -euo pipefail

cd /home/abaumgartner/CRISP
source ~/crisp_env/bin/activate

ITER=$(cat checkpoints/hf_weights/iteration.txt 2>/dev/null || echo "unknown")
echo "=== CRISP Eval: alice_hf (iter $ITER) ==="
echo "Date: $(date)"

python scripts/eval_model.py \
    --model checkpoints/hf_weights/alice_hf \
    --n-problems 500 \
    --output "trained_iter${ITER}.json"

echo ""
echo "=== DONE at $(date) ==="
