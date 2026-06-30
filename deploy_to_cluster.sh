#!/bin/bash
# Deploy the CANDI attention experiment code to the cluster.
#
# Usage:
#   ./deploy_to_cluster.sh [CLUSTER_LOGIN]
#
# Example:
#   ./deploy_to_cluster.sh ar569@dcc-login.oit.duke.edu
#
# This script:
#   1. Packages the necessary code (no checkpoint — that's too large)
#   2. Syncs it to the cluster at /work/ar569/candi-attention-exp/
#   3. Prints instructions for submitting the job

set -euo pipefail

CLUSTER_LOGIN="${1:-ar569@dcc-login.oit.duke.edu}"
REMOTE_DIR="/work/ar569/candi-attention-exp"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Packaging code for cluster ==="

# Files to send (exclude large checkpoint, generated data, __pycache__)
INCLUDE_FILES=(
  run_attention_exp.py
  analyze_attention.py
  slurm_attention_exp.sbatch
  algo.py
  trainer_base.py
  dataloader.py
  metrics.py
  text_metrics.py
  models/__init__.py
  models/dit.py
  models/dit_cont.py
  models/ema.py
  denoise_diff/__init__.py
  denoise_diff/model.py
  denoise_diff/harness.py
  denoise_diff/metrics.py
  denoise_diff/attention_hooks.py
  configs/
)

echo "Syncing to ${CLUSTER_LOGIN}:${REMOTE_DIR}/ ..."

# Use rsync for efficient transfer
rsync -avz --progress \
  --include='configs/***' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='experiments/out/' \
  --exclude='gen_imgs/' \
  --exclude='basecheckpoint/' \
  --exclude='.git/' \
  --exclude='wandb/' \
  "${INCLUDE_FILES[@]/#/$SCRIPT_DIR/}" \
  "${CLUSTER_LOGIN}:${REMOTE_DIR}/" 2>/dev/null || {
    # If rsync with individual files fails, fall back to directory sync
    echo "Falling back to full directory rsync..."
    rsync -avz --progress \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='experiments/out/' \
      --exclude='gen_imgs/' \
      --exclude='basecheckpoint/*.ckpt' \
      --exclude='.git/' \
      --exclude='wandb/' \
      . "${CLUSTER_LOGIN}:${REMOTE_DIR}/"
}

echo
echo "=== Deployed to ${CLUSTER_LOGIN}:${REMOTE_DIR}/ ==="
echo
echo "Next steps on the cluster:"
echo "  1. SSH in:  ssh ${CLUSTER_LOGIN}"
echo "  2. Check/create the conda env (needs torch, flash-attn, einops, omegaconf, datasets, matplotlib):"
echo "       conda create -n candienv python=3.11"
echo "       conda activate candienv"
echo "       pip install torch flash-attn einops omegaconf datasets matplotlib huggingface_hub"
echo "  3. Copy the checkpoint:"
echo "       cp <path-to-candi-last.ckpt> ${REMOTE_DIR}/basecheckpoint/candi-last.ckpt"
echo "  4. Submit the job:"
echo "       cd ${REMOTE_DIR}"
echo "       sbatch slurm_attention_exp.sbatch"
echo "  5. Monitor:"
echo "       squeue -u \$USER"
echo "       tail -f logs/candi_attn_exp-<JOBID>.out"
