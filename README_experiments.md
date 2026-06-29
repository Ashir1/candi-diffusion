# Denoise-vs-Difficulty Experiments

Read-only probes of the trained CANDI model that study how **denoising order**, **token
difficulty** (predictive entropy), **rescue**, and **token importance** relate. Nothing in the
core CANDI code is modified (only `trainer_base.py`'s `hydra.utils` import was made lazy so the
model loads without hydra).

## Layout

```
denoise_diff/            # the reusable package
  metrics.py     # pure functions: entropy, gold-NLL, top-k accuracy, quantile buckets, reveal sets
  harness.py     # drive the unmodified model: trace_sample / denoiser_probe / gold_reconstruct_trace / gold_renoise_trace
  model.py       # load_model (from ckpt, no hydra), load_gold (text), gen_ppl, free
  plotting.py    # shared figure helpers (save_fig, grouped_bars, colours, corr, load)
run_experiments.py       # CLI: collect data  ->  experiments/out/<run>/*.pt
analyze.py               # CLI: make figures  ->  gen_imgs/<run>/*.png
```

To add an experiment: add a `run_expN` in `run_experiments.py` (using `harness`), a view in
`analyze.py` (using `plotting`), and a row to the table below.

## The experiments

| exp | question | data file | `analyze.py` view → figures |
|---|---|---|---|
| 1 | is reveal **order** independent of difficulty? | `exp1_2_trace.pt` | `standard` → `exp1_order_vs_difficulty.png`; `per_ratio`; `difftime` |
| 2 | **entropy collapse** of still-masked tokens, by difficulty | `exp1_2_trace.pt` | `standard` → `exp2_entropy_collapse.png`; `per_ratio`; `difftime` |
| 3 | which **context** (near/far/left/right) collapses entropy | `exp3_structured.pt` | `standard`/`per_ratio`/`difftime` → `exp3_*` |
| 4 | **rescue + accuracy** vs gold (top-1/top-3) | `exp4_rescue.pt` | `standard` → `exp4_rescue_{curves,taxonomy}.png`; `per_ratio`; `difftime` (+ forward) |
| 6 | **more reveal-time rescues hard tokens** (gold) | `exp6_recon.pt` | `exp6` → `exp6_{rescue,time_vs_rescue}_diff{R}.png` |
| 7 | **re-noise & resample** importance (gold, in-distribution) | `exp7_renoise.pt` | `exp7` → `exp7_renoise_diff{R}.png` |

(Exp 5 — out-of-distribution perturbation — was removed; exp 7 is its clean in-distribution successor.)

**Difficulty** = the model's predictive entropy at a token (2506.01939). "easy/hard" buckets split
by that entropy; `difftime` re-buckets by entropy measured at 10/20/30/40 % denoised instead of 0 %.

## Run it

Inside the candi env (torch + flash-attn + omegaconf), on a GPU:

```bash
# collect data (exp 4/6/7 need gold text; wikitext auto-downloads ~5 MB)
python run_experiments.py --exp all --out experiments/out/run_big \
  --n-seqs 64 --steps 256 --batch-size 4 --gold-hf wikitext:wikitext-2-raw-v1:validation --gen-ppl

# make all figures
python analyze.py --in experiments/out/run_big --figdir gen_imgs/run_big --diff-ratios 10,20,30,40
```

Run a subset with `--exp 1,2,3` or a single view with `--views exp7`.

## Notes & gotchas

- **Checkpoint** (default in `--ckpt`): `…/candi-last.ckpt` — `model=small`, length 1024, gpt2/OWT.
  Loaded from the config saved *inside* the ckpt (omegaconf only); `_orig_mod` keys dropped, EMA ignored.
- **`--batch-size` must be ≤ 4** at length 1024 — the soft `(B,L,V)` double-precision tensors OOM at 8 on 24 GB.
- **Gold sources** (exp 4/6/7): `--gold-hf NAME[:CONFIG[:SPLIT]]`, `--gold-file some.txt`, or OpenWebText
  via `--scratch-dir` (a dir holding `owt/`). Exp 1/2/3 need no data.
- **`--gen-ppl`** (exp 7) downloads gpt2-large (~3 GB) and computes generative perplexity; without it
  exp 7 still records entropy/accuracy trajectories.
- **Resumable**: exp 7 saves after each (ratio, condition) and skips finished ones on restart.
- Design rationale & caveats: [denoise_vs_difficulty_plan.md](denoise_vs_difficulty_plan.md).
