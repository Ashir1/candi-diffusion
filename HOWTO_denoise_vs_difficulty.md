# HOWTO — Run the Denoise-vs-Difficulty Experiments

This implements [`denoise_vs_difficulty_plan.md`](denoise_vs_difficulty_plan.md). It probes the
**existing, unmodified** CANDI sampler to relate *denoising order*, *token difficulty* (predictive
entropy), and *output quality*, across fixed reveal ratios `{0,20,40,60,80}%`.

> Nothing in `algo.py` / the model is changed. The harness re-implements the non-cached sampling
> loop and adds logging + a probe + a perturbation, all by **calling existing model methods**.

---

## 1. Files

| file | role |
|---|---|
| [`difficulty.py`](difficulty.py) | Pure metrics: predictive entropy, gold-NLL, accuracy, top-k, rescue / false-rescue, reveal-set builders. Swap difficulty notions here. |
| [`denoise_harness.py`](denoise_harness.py) | `trace_sample` (H1 trajectory), `denoiser_probe` (H2 single forward), `perturb_and_continue` (exp 5). |
| [`scripts/run_denoise_entropy_experiment.py`](scripts/run_denoise_entropy_experiment.py) | Loads the checkpoint, runs exp 1–5, dumps tensors to `--out`. |
| [`scripts/analyze_denoise_entropy.py`](scripts/analyze_denoise_entropy.py) | Reads the dumps, writes plots + stats to `--figdir`. |

Which experiment uses which harness (see plan §3):

| exp | what | harness | needs gold? |
|---|---|---|---|
| 1 | reveal order ⊥ difficulty (committed vs masked cohort) | H1 trace | no |
| 2 | entropy-collapse rate, easy vs hard | H1 trace (same dump as exp 1) | no |
| 3 | which context collapses entropy (near/far, left/right) | H2 probe on a self-generated sample | no |
| 4 | rescue / false-rescue taxonomy | H2 probe on **real validation data** | **yes** |
| 5 | perturb hardest/random/easiest committed tokens → quality | perturb-and-continue | no (gen-PPL is reference-free) |

---

## 2. Environment

Run inside the **candi env** on a **GPU**. Needed: `torch`, `transformers`, `lightning`,
`torchmetrics`, `flash-attn` (the `dit_cont.py` backbone imports `flash_attn` at module load), and
**`omegaconf`**.

**Hydra is NOT required** for these scripts. `trainer_base.py`'s `hydra.utils` import was made lazy
(training-only), and the driver reads the model config **straight from the checkpoint**, so you only
need `omegaconf` (which the checkpoint's pickled config requires anyway):

```bash
conda activate <candi-env>
pip install omegaconf          # if missing -- hydra-core is NOT needed
cd /home/ashir/candi-diffusion
```

> `ashirenv` has torch but no flash-attn — it can run `difficulty.py` + the harness *imports* + the
> mock tests, but not the real model.

Checkpoint (already located): `/data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt`
(`model=small`, 768-d / 12-layer, length 1024, gpt2 / OpenWebText). The driver loads the **trained
config from the checkpoint**, so the trained knobs (`step_size=0.5`, `sigma_min/max=0.1/4.0`) apply
automatically; any newer algo keys missing from the older saved schema are topped up from
`configs/algo/candi.yaml`, and the `_orig_mod` (torch.compile) state-dict keys are dropped.

---

## 3. Run everything

```bash
python scripts/run_denoise_entropy_experiment.py --exp all \
  --ckpt /data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt \
  --out experiments/out/run1 \
  --n-seqs 32 --steps 256 --batch-size 8 \
  --ratios 0,20,40,60,80 --perturb-ratios 20,40,60,80 \
  --scratch-dir /path/to/owt_cache_parent     # only needed for exp 4 (OWT validation data)
```

Then plot:

```bash
python scripts/analyze_denoise_entropy.py --in experiments/out/run1 --figdir gen_imgs/denoise
```

### Run a single experiment
`--exp` takes a comma list, e.g. `--exp 1,2` (one trace serves both), `--exp 3`, `--exp 4`, `--exp 5`.

### Exp 4 gold data (no OWT needed)
Exp 4 is the only one that needs real "gold" sequences. Easiest is a tiny HF dataset (auto-downloads ~5 MB):
```bash
python scripts/run_denoise_entropy_experiment.py --exp 4 \
  --ckpt /data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt \
  --out experiments/out/run1 --n-seqs 32 \
  --gold-hf wikitext:wikitext-2-raw-v1:validation
```
Alternatives: `--gold-file some.txt` (any UTF-8 text), or the default OpenWebText path (`--scratch-dir`
pointing at a dir holding the `owt/` cache; run `bash manual_download.sh` first if you don't have it).

### Exp 5 generative perplexity (optional)
`--gen-ppl` makes `transformers` download **gpt2-large (~3 GB)** to the HF cache on first use (needs
internet) and computes generative perplexity. Without it, exp 5 reports only the reference-free
**downstream drift** (cheap, no download). To pre-cache the model:
`python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('gpt2-large')"`.

### Quick smoke run (tiny, fast)
```bash
python scripts/run_denoise_entropy_experiment.py --exp 1,2 --n-seqs 4 --steps 64 --batch-size 4 \
  --out experiments/out/smoke
python scripts/analyze_denoise_entropy.py --in experiments/out/smoke
```

---

## 4. Outputs

Dumps in `--out`:

| file | contents |
|---|---|
| `exp1_2_trace.pt` | `H_traj (S,N,L)` entropy field, `reveal_step (N,L)`, `committed (N,L)`, `H0 (N,L)`, `final_tokens` |
| `exp3_structured.pt` | per-target entropy `ent_r{R}_{mode}` for modes near/far/left/right/balanced, `targets` |
| `exp4_rescue.pt` | `entropy/gold_nll/correct (R,N,L)`, `target_mask` (positions kept masked across ratios) |
| `exp5_perturb.pt` | per (ratio, condition) `final_tokens`, `perturb_mask`, optional `gen_ppl` |
| `meta.json` | the exact args/ratios used |

Figures in `--figdir` (default `gen_imgs/denoise/`):

| figure | shows |
|---|---|
| `exp1_order_vs_difficulty.png` | reveal-frac vs entropy-at-reveal (Spearman/Pearson) **+ the H0 null control** (should be flat) + committed−cohort per step |
| `exp2_entropy_collapse.png` | entropy-collapse curves by initial-difficulty quartile |
| `exp3_structured_context.png` | entropy vs ratio for near/far/left/right, faceted by difficulty quartile |
| `exp4_rescue_curves.png` | entropy & gold-NLL collapse, hard vs easy targets |
| `exp4_rescue_taxonomy.png` | entropy-rescue vs NLL-rescue scatter (false rescue in red) + the 2×2×2 taxonomy bars |
| `exp5_perturbation_importance.png` | downstream drift (and gen-PPL) for hard/random/easy perturbations |

---

## 5. Key knobs

| flag | default | note |
|---|---|---|
| `--n-seqs` | 32 | sequences traced/probed; raise for stable correlations (200–500) |
| `--steps` | 256 | sampler steps; finer = better reveal-order resolution, more memory (`H_traj` is `S×N×L` fp16) |
| `--batch-size` | 8 | lower if OOM |
| `--ratios` | `0,20,40,60,80` | reveal ladder for exp 3/4 |
| `--perturb-ratios` | `20,40,60,80` | exp 5 (no `0` — nothing committed to perturb) |
| `--n-targets` | 8 | exp 3 target positions per reference sequence |
| `--target-frac` | 0.2 | exp 4 fraction of positions kept masked & tracked across ratios |
| `--frac` | 0.2 | exp 5 fraction of **committed** tokens perturbed per condition |
| `--gen-ppl` | off | exp 5: compute GPT-2-large generative perplexity (downloads ~3 GB). Off ⇒ only downstream drift. |
| `--gold-hf` / `--gold-file` | — | exp 4 gold source (HF dataset `NAME[:CONFIG[:SPLIT]]`, or a text file); else OpenWebText via `--scratch-dir` |

---

## 6. Interpreting results (pointers to the plan)

- **Exp 1** — expect Spearman ≈ 0 and the per-step committed−cohort diff ≈ 0 ⇒ reveal order is independent of difficulty (plan §1.1). The **H0 null control** should be flat; the `reveal-frac vs entropy-at-reveal` panel will trend down, but that's the *conditioning artifact* (plan §1.2/§2), not selection.
- **Exp 2** — easy quartile flat-low; hard quartile high then (often bimodal) collapse.
- **Exp 3** — does right-only context help? (diffusion-specific; an AR model couldn't use it.) Do hard tokens need far/specific context vs local for easy?
- **Exp 4** — the only correctness experiment: rescue (entropy & gold-NLL), **false rescue** (entropy down but confident in the wrong token), and the hard/easy × rescued/not × correct/wrong cells.
- **Exp 5** — if hardest-token perturbation drifts/degrades more than random/easy, initially-hard tokens are more pivotal (the 80/20 hypothesis); if less, they're "free slots."

---

## 7. Troubleshooting

- **`ModuleNotFoundError: flash_attn`** → wrong env; use the candi env (§2). **`No module named 'omegaconf'`** → `pip install omegaconf` (hydra-core is NOT needed).
- **Checkpoint load** → `_orig_mod` (torch.compile) keys are dropped and missing newer algo keys are topped up automatically. EMA weights are present but ignored (`model.ema=None`); apply the EMA shadow before eval to use them.
- **Exp 4 can't find data** → easiest is `--gold-hf wikitext:wikitext-2-raw-v1:validation` (small auto-download) or `--gold-file <txt>`; the OWT path needs `--scratch-dir` holding `owt/`. Exp 1/2/3/5 need no data.
- **OOM** → lower `--batch-size`, `--steps`, `--n-seqs`, or `--n-targets`.
- **gen-PPL slow/failing** → it downloads gpt2-large; drop `--gen-ppl` to rely on downstream drift only.

---

## 8. Validation status

`difficulty.py` is unit-tested and the **harness + driver + analysis** are integration-tested against a
mock model that mimics the CANDI interface (shapes, reveal monotonicity, entropy-drops-with-context,
all 6 figures generated). End-to-end on the **real** checkpoint still needs the candi env + GPU — start
with the smoke run in §3.
