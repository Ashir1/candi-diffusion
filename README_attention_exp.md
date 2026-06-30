# Attention Evolution Experiment

**Question:** How does the attention that hard tokens pay to each other evolve over the diffusion denoising process?

## Files Created

| File | Purpose |
|------|---------|
| `run_attention_exp.py` | Main experiment script — gold-text reconstruction with attention capture |
| `analyze_attention.py` | Analysis & plotting script — generates figures from results |
| `denoise_diff/attention_hooks.py` | Hook utility that captures attention weights from DDiTBlock layers |
| `slurm_attention_exp.sbatch` | SLURM batch script for the Duke cluster |
| `deploy_to_cluster.sh` | Deploys code to the cluster via rsync |
| `receive_results.py` | Simple HTTP server on EC2 to receive results back |

## Quick Start (Cluster)

### 1. Deploy code to cluster

```bash
# From this EC2 instance (or your local machine)
./deploy_to_cluster.sh ar569@dcc-login.oit.duke.edu
```

### 2. Set up the environment on the cluster

```bash
ssh ar569@dcc-login.oit.duke.edu
cd /work/ar569/candi-attention-exp

# Create conda env (one-time)
conda create -n candienv python=3.11 -y
conda activate candienv
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install einops omegaconf datasets matplotlib huggingface_hub transformers

# Copy the checkpoint
mkdir -p basecheckpoint
cp /path/to/candi-last.ckpt basecheckpoint/candi-last.ckpt
```

### 3. Submit the job

```bash
# Default: 32 sequences, 256 steps, layers 0/5/11, capture every 16 steps
sbatch slurm_attention_exp.sbatch

# Or customize:
N_SEQS=64 STEPS=256 BATCH_SIZE=8 sbatch slurm_attention_exp.sbatch

# Smoke test (quick):
N_SEQS=2 STEPS=32 CAPTURE_EVERY=8 BATCH_SIZE=1 sbatch slurm_attention_exp.sbatch
```

### 4. Monitor

```bash
squeue -u $USER
tail -f logs/candi_attn_exp-<JOBID>.out
```

## Getting Results Back

### Option A: SCP directly

```bash
# From EC2 or local machine
scp -r ar569@dcc-login.oit.duke.edu:/work/ar569/candi-attention-exp/experiments/out/attn_run_* \
    /home/ec2-user/together/ashir/candi-diffusion/experiments/out/
```

### Option B: Auto-upload from SLURM job

Set these env vars before submitting:
```bash
EC2_HOST=<your-ec2-public-ip> EC2_KEY=/path/to/key.pem sbatch slurm_attention_exp.sbatch
```

### Option C: HTTP upload server on EC2

```bash
# On EC2 (keep running):
python3 receive_results.py --port 8432

# From cluster (in the sbatch or manually):
tar -czf /tmp/results.tar.gz experiments/out/attn_run_* gen_imgs/attention_evolution/
curl -X POST http://<EC2_IP>:8432/upload --data-binary @/tmp/results.tar.gz
```

## Analyzing Results Locally

Once you have `attn_evolution.pt` on this instance:

```bash
python3.11 analyze_attention.py \
  --data experiments/out/attn_run_XXXX/attn_evolution.pt \
  --out gen_imgs/attention_evolution/
```

This generates 4 plots:
- `hard_token_attention_timeseries.png` — Where hard tokens attend over time
- `all_pairs_attention_timeseries.png` — Full group×group matrix over time
- `attention_matrix_snapshots.png` — Heatmaps at early/mid/late
- `hard_hard_attention_focus.png` — Hard→Hard vs uniform baseline

## What If Cursor/EC2 Gets Stuck

1. **If a pip install hangs**: The machine is OOM compiling flash-attn. Kill it with `kill -9 <PID>` or reboot.
2. **To avoid crashes**: Don't compile flash-attn on this instance. Run experiments on the cluster only.
3. **If you need to reboot**: `sudo reboot` — Cursor will reconnect automatically.
4. **Analysis only (no GPU needed)**: `analyze_attention.py` runs fine on CPU with just torch + matplotlib.
