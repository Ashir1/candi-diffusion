#!/bin/bash
#SBATCH --job-name=candi_attn_exp
#SBATCH --account=wengerlab
#SBATCH --partition=wengerlab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:6000_ada:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# -------------------------
# Project directory
# -------------------------
WORK_DIR=/work/ar569/candi-attention-exp
cd "$WORK_DIR"
mkdir -p logs experiments/out

# -------------------------
# Experiment parameters (override via environment)
# -------------------------
N_SEQS=${N_SEQS:-32}
STEPS=${STEPS:-256}
BATCH_SIZE=${BATCH_SIZE:-4}
LAYERS=${LAYERS:-"0,5,11"}
CAPTURE_EVERY=${CAPTURE_EVERY:-16}
N_GROUPS=${N_GROUPS:-4}
GOLD_SOURCE=${GOLD_SOURCE:-"wikitext:wikitext-2-raw-v1:validation"}
OUT_NAME=${OUT_NAME:-"attn_run_$(date +%Y%m%d_%H%M%S)"}
OUT_DIR="experiments/out/$OUT_NAME"

# EC2 upload target (set to empty to skip upload)
EC2_HOST=${EC2_HOST:-""}
EC2_USER=${EC2_USER:-"ec2-user"}
EC2_DEST=${EC2_DEST:-"/home/ec2-user/together/ashir/candi-diffusion/experiments/out/"}
EC2_KEY=${EC2_KEY:-""}

# -------------------------
# Cache/temp paths on /work
# -------------------------
export HF_HOME=/work/ar569/huggingface
export HF_HUB_CACHE=/work/ar569/huggingface/hub
export HF_DATASETS_CACHE=/work/ar569/huggingface/datasets
export HF_XET_CACHE=/work/ar569/huggingface/xet
export TMPDIR=/work/ar569/tmp
export PIP_CACHE_DIR=/work/ar569/pip-cache
export CONDA_PKGS_DIRS=/work/ar569/conda-pkgs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$HF_XET_CACHE" "$TMPDIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS"

# -------------------------
# Conda environment
# -------------------------
set +u
source ~/.bashrc
conda activate candienv
set -u

# -------------------------
# Debug info
# -------------------------
echo "=== Slurm ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURMD_NODENAME"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"

echo
echo "=== Python ==="
which python
python --version
python -c 'import torch; print("torch", torch.__version__); print("CUDA", torch.cuda.is_available(), torch.cuda.device_count()); import flash_attn; print("flash_attn", flash_attn.__version__)'

echo
echo "=== GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

echo
echo "=== Experiment Config ==="
echo "N_SEQS: $N_SEQS"
echo "STEPS: $STEPS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "LAYERS: $LAYERS"
echo "CAPTURE_EVERY: $CAPTURE_EVERY"
echo "N_GROUPS: $N_GROUPS"
echo "GOLD_SOURCE: $GOLD_SOURCE"
echo "OUT_DIR: $OUT_DIR"

# -------------------------
# Required files
# -------------------------
CKPT="basecheckpoint/candi-last.ckpt"
test -f "$CKPT" || { echo "MISSING checkpoint: $CKPT"; exit 1; }
echo "OK checkpoint: $CKPT"

# -------------------------
# Run experiment
# -------------------------
echo
echo "=== Starting attention evolution experiment ==="

python -u run_attention_exp.py \
  --ckpt "$CKPT" \
  --out "$OUT_DIR" \
  --n-seqs "$N_SEQS" \
  --steps "$STEPS" \
  --batch-size "$BATCH_SIZE" \
  --layers "$LAYERS" \
  --capture-every "$CAPTURE_EVERY" \
  --n-groups "$N_GROUPS" \
  --gold-hf "$GOLD_SOURCE" \
  --seed 42

echo
echo "=== Experiment complete ==="
echo "Results saved to: $OUT_DIR"
ls -lh "$OUT_DIR/"

# -------------------------
# Generate analysis plots
# -------------------------
echo
echo "=== Generating plots ==="
PLOT_DIR="gen_imgs/attention_evolution/$OUT_NAME"
python -u analyze_attention.py \
  --data "$OUT_DIR/attn_evolution.pt" \
  --out "$PLOT_DIR"
echo "Plots saved to: $PLOT_DIR"

# -------------------------
# Upload results to EC2 (optional)
# -------------------------
if [[ -n "$EC2_HOST" ]]; then
  echo
  echo "=== Uploading results to EC2 ==="
  SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
  if [[ -n "$EC2_KEY" ]]; then
    SSH_OPTS="$SSH_OPTS -i $EC2_KEY"
  fi

  # Create tarball of results + plots
  TARBALL="$TMPDIR/attn_results_${OUT_NAME}.tar.gz"
  tar -czf "$TARBALL" "$OUT_DIR" "$PLOT_DIR"

  scp $SSH_OPTS "$TARBALL" "${EC2_USER}@${EC2_HOST}:${EC2_DEST}/"
  if [[ $? -eq 0 ]]; then
    echo "Upload successful: $TARBALL -> ${EC2_HOST}:${EC2_DEST}/"
    # Also send a completion signal
    ssh $SSH_OPTS "${EC2_USER}@${EC2_HOST}" \
      "cd ${EC2_DEST} && tar -xzf attn_results_${OUT_NAME}.tar.gz && rm attn_results_${OUT_NAME}.tar.gz && echo 'Extracted on EC2'"
  else
    echo "WARNING: Upload failed. Results are still saved locally at: $OUT_DIR"
  fi
else
  echo
  echo "EC2_HOST not set — skipping upload. Retrieve results manually:"
  echo "  scp $SLURMD_NODENAME:$WORK_DIR/$OUT_DIR/attn_evolution.pt <local_path>"
fi

echo
echo "=== Done ==="
