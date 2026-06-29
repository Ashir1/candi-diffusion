# Experiment Plan — Denoising Order vs. Token Difficulty in CANDI

**Goal.** Probe **the existing CANDI sampler, with no algorithmic modifications**, to understand the relationship between *denoising order*, *token difficulty*, and *final output quality*. The study is a suite of **5 experiments evaluated at a fixed set of "denoised token ratios"** `r ∈ {0, 20, 40, 60, 80}%` (experiment 5 uses `{20, 40, 60, 80}%`).

> The two cited papers ([2506.01939](https://arxiv.org/abs/2506.01939) "80/20 high-entropy tokens", [2506.02281](https://arxiv.org/abs/2506.02281) "Angles Don't Lie") are RL-for-reasoning papers; we port their *notions of difficulty* (predictive entropy; optionally angle concentration) into masked-diffusion sampling. Definitions reproduced in §2.

> **Scope decision (current run):** we test CANDI as-is. We do **not** modify the reveal rule, do **not** add confidence-ordered decoding, and **accept** the confound that in CANDI a masked token's *continuous* channel leaks a noisy embedding of its own identity (see §1.2) — i.e. when more discrete context is revealed, the still-masked token's own continuous noise also drops. Disentangling those two information channels is explicitly **out of scope for now** (noted in §6 as a future control).

---

## 0. Prerequisites & confirmed setup

**Checkpoint (✅ found):** `/data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt` (~2.7 GB; epoch 38, step 641,492; EMA weights included).

**Architecture / data — read from the checkpoint's saved config (`model=small`):**

| field | value |
|---|---|
| backbone | `cont_dit` (CANDI), bidirectional, `causal_attention=false` |
| hidden_size / n_blocks / n_heads | 768 / 12 / 12 |
| length | 1024 |
| tokenizer / data | gpt2 / OpenWebText (`openwebtext-train` / `-valid`) |
| output vocab | 50259 (real vocab + mask column) |
| trained algo knobs | `mixed_coeff=0.5`, **`step_size=0.5`**, `max/min_percentile=0.25/0.01`, `use_percentile_scheduling=true`, **`sigma_min/max=0.1/4.0`**, `top_k=100` |

> ⚠️ The repo's `configs/algo/candi.yaml` differs from the **trained** config on `step_size` (1.0 vs **0.5**) and `sigma_min/max` (0.01/2.0 vs **0.1/4.0**). `main.py` loads weights against a *runtime* Hydra config, so **override these at launch to match training**, or sampling drifts from how the model behaved in training.

| Item | Status | Action |
|---|---|---|
| Checkpoint | ✅ found (path above) | — |
| Project env (torch + hydra + flash-attn) | **verify** | `ashirenv` has torch 2.8 but **no** hydra/omegaconf/flash-attn; repo pins torch 2.3 + `flash-attn==2.6.1` (imported by `dit_cont.py`). Confirm/create the candi env before running. |
| Loading caveat | — | state_dict has both `backbone.*` and `backbone._orig_mod.*` keys (torch.compile artifact) → `load_from_checkpoint` may need `_orig_mod` stripped or `strict=False`. Decide `eval.disable_ema` (EMA weights present). |
| Validation data (gold) | needed **exp 4 only** | `data=openwebtext-split`, gpt2 tokenizer. |

**Sanity check first** — confirm coherent text before building anything:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py mode=sample_eval data=openwebtext-split model=small algo=candi \
  model.length=1024 algo.step_size=0.5 algo.sigma_min=0.1 algo.sigma_max=4.0 \
  sampling.steps=256 loader.eval_batch_size=8 sampling.num_sample_batches=1 \
  eval.checkpoint_path=/data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt \
  +wandb.offline=true
```
For clean per-step tracing in the experiments add `algo.sampler=loop` (non-cached path; exposes `p_x0` / `clean_mask` transitions directly).

---

## 1. How CANDI sampling works (grounded, read-only facts)

Default sampler [`generate_samples_nocache`](algo.py#L903) denoises **all positions in parallel each step** with **full bidirectional attention** (`causal=False`, [dit_cont.py:452](models/dit_cont.py#L452), [dit_cont.py:140](models/dit_cont.py#L140)). Per step `i`:
```
x_cont, p_x0 = self._continuous_step(...)    # denoiser forward; p_x0 = predicted clean dist over vocab, ALL positions
x, clean_mask = self._discrete_step(...)     # commits (freezes) a random subset of still-masked positions
```
- **`p_x0`** ([algo.py:1077](algo.py#L1077)) — per-position distribution over the real vocab → take **entropy** of it. Available every step for every position.
- **`clean_mask` (B,L)** — boolean "already committed". A position's **reveal step τ** = the iteration it flips 0→1. Monotone (once clean, stays clean), and committed values are **frozen** ([algo.py:1052](algo.py#L1052)).
- **Global reveal ratio** at step `i` = `clean_mask.float().mean()`. We snapshot experiments at the steps where this crosses `r ∈ {0,20,40,60,80}%`.

### 1.1 Reveal order is random (the premise to validate)
Per-position commit probability is `(t−s)/t`, **identical for all positions, independent of `p_x0`/entropy** ([algo.py:1037-1042](algo.py#L1037-L1042) and [algo.py:1023](algo.py#L1023)). So intrinsic difficulty and τ are independent *by construction*. **Experiment 1 verifies this empirically.** (The token *value* written on commit is sampled from `p_x0`; only the *whether-to-commit* is content-blind.)

### 1.2 A masked token is not information-free (accepted confound)
In [`q_xt`](algo.py#L837-L839), masked positions carry `onehot(gold) + continuous_noise·randn` — a **noisy continuous embedding of the true token** — mixed 50/50 with a learned mask embedding ([dit_cont.py:523](models/dit_cont.py#L523)). `continuous_noise` is tied to the global ratio. So raising `r` both (a) adds revealed neighbor context **and** (b) lowers the masked token's own continuous noise. **We accept (a)+(b) entangled for this run** (§6 future control).

### 1.3 Measure entropy *before* the commit
Already-clean positions are forced to one-hot ([algo.py:748](algo.py#L748)) ⇒ entropy ≈ 0 afterward. At iteration `i`, `_continuous_step` runs with `clean_mask` *before* this step's commit, so a position revealed at `i` still has a genuine `p_x0[i, pos]`. **Capture entropy/NLL at the pre-commit prediction.**

---

## 2. Definitions

### Difficulty (entropy, 2506.01939 Eq.1)
`H_i(pos) = − Σ_v p_x0[i,pos,v] · log p_x0[i,pos,v]` (natural log / nats; over the real vocab). Their high-entropy "forking" cutoff ≈ 0.672 nats (80th pct).
- **Initial difficulty** `D_i := H(pos)` measured at the starting ratio (default `r=0`, fully masked). Reference-free.
- Optional gold-NLL difficulty: `D^{NLL}_i := −log p_x0[pos, gold_i]` at `r=0` (needs ground truth).

### Gold-token NLL (ground truth — **experiment 4 only**)
`NLL_i(r) = − log p_x0[pos, gold_i]` evaluated with `r%` context revealed and `pos` masked.

### Accuracy (ground truth — **experiment 4 only**)
`A_i = 1[ argmax_v p_x0[pos,v] == gold_i ]`. For open text, also report **top-k accuracy** and prefer NLL — a single gold token undercounts legitimate lexical variation.

### Rescue (both notions) and False rescue — **experiment 4**
For a position kept masked while context grows from an **early** ratio to a **late** ratio:
- **Entropy rescue:** `Rescue^H_i = H_i(early) − H_i(late)` (uncertainty dropped).
- **Gold-NLL rescue (preferred):** `Rescue^{NLL}_i = NLL_i(early) − NLL_i(late)` (confidence in the *correct* token grew).
- **False rescue:** entropy fell **but** the model became confident in the **wrong** token — operationally `H_i(late)` low (e.g. < forking cutoff) **and** `NLL_i(late)` high (e.g. above a percentile). I.e. `Rescue^H_i > 0` while `Rescue^{NLL}_i ≤ 0`. This cell is itself a finding.
- A **rescued** token: high `D_i`, then low `H_i(late)`/`NLL_i(late)`, and `A_i = 1`.

> Default early/late = `r=0` (or 20) → `r=80`. Report rescue across the full ratio ladder, not just endpoints.

### (Optional) Difficulty B — angle concentration (2506.02281 Eq.5)
Average pairwise cosine similarity of token hidden states (higher ⇒ easier); hidden states available via `ret_embedding=True` ([dit_cont.py:538](models/dit_cont.py#L538)). Cleanly separable add-on; defer until the entropy story is in.

---

## 3. Two evaluation harnesses (both use unmodified CANDI)

Only **exp 4** needs ground-truth *target* tokens (correctness). Everything else is **reference-free** — entropy is computed from `p_x0` alone.

- **H1 — Sampler-trajectory logging (reference-free).** Run `generate_samples_nocache` unchanged; log per step: `p_x0` (→ per-position entropy on the fly, store scalars only), `clean_mask` before/after, committed ids. Bucket by global reveal ratio. **Used by exp 1, exp 2, and exp 5** (exp 5 adds a perturb-and-continue intervention on top). Exp 2 just reads the entropy of still-masked positions as the sampler's own random reveal grows the context — no probe needed.
- **H2 — Fixed-ratio reveal probe (single forward, constructed reveal set).** Take a **reference sequence** `x_ref`, construct `xt` revealing a chosen `r%` subset (clean) with the rest masked at the schedule's continuous noise, run **one** `model.forward`, read `p_x0` for masked positions. **The reveal *set* is the controlled variable.** Two uses:
  - **exp 3** — reference is a **self-generated CANDI sample** (or any fixed sequence); structured reveal sets (near/far, left/right); **entropy only, reference-free** (no gold targets).
  - **exp 4** — reference is **real validation data**, so `gold_i` is known and we additionally compute gold-NLL / accuracy / rescue. *This is the only place ground-truth targets enter.*

**Memory:** never store full `p_x0` (B,L,V) across steps — compute entropy/NLL on the fly and store scalars `(steps or ratios) × L`. (Full distribution × steps is hundreds of GB at L=1024.)

---

## 4. The five experiments

Ratio ladder `R = {0,20,40,60,80}%` for 1–4; `R\{0} = {20,40,60,80}%` for 5.

### Exp 1 — Is reveal order independent of difficulty? (harness H1)
**Setup.** Run the real sampler. At each commit event, compare the **entropy of the just-committed token(s)** to the **mean entropy of the still-masked cohort** at that same step; bucket by current ratio.
**Metric.** Per bucket: `mean(H_committed) − mean(H_masked_cohort)`; also each committed token's **rank** within the cohort entropy distribution (test rank ~ Uniform via KS/permutation — tighter than the mean test).
**Predicted.** ≈ 0 with no systematic sign / uniform ranks ⇒ confirms order ⊥ intrinsic difficulty (§1.1).
**Pitfalls.** Filter pinned positions (BOS via `ignore_bos`, prompt tokens — τ=0, trivially low entropy).

### Exp 2 — Entropy-collapse rate, easy vs hard (harness H1, reference-free)
**Setup.** From the sampler trajectory, for each position **still masked** at ratio `r`, record its entropy `H_i(r)`. Group positions by `D_i` (initial entropy at `r≈0`, e.g. quartiles). No ground truth, no probe — the sampler's own random reveal supplies the growing context. (Average over many sequences/seeds for stable curves.)
**Metric.** Entropy-collapse curves `H_i(r)` vs `r ∈ R`, per difficulty group.
**Predicted.** Easy = flat-low; hard = high-start, then a **bimodal** split (rescuable: collapses; persistent: stays high).
**Pitfalls.** A token leaves the "still-masked" set once it commits, so late-ratio curves condition on "survived to `r`" — but reveal is random (exp 1), so that survivorship is difficulty-independent. Accepted confound §1.2 (collapse mixes neighbor-context + own-continuous-noise drop). *Correctness of the collapse is deliberately deferred to exp 4.*

### Exp 3 — Which context drives collapse: near/far, left/right (harness H2, reference-free)
**Setup.** Probe a **fixed reference sequence** (a self-generated CANDI sample is fine — no gold targets needed): for target token `i` kept masked, reveal a **structured** subset of the *other* tokens at matched count — (a) nearby window around `i` vs (b) far/spread; (c) left-only (`<i`) vs (d) right-only (`>i`) vs (e) balanced — and measure `H_i`.
**Metric.** Entropy `H_i` collapse per condition × difficulty group.
**Predicted.** Easy/syntactic tokens rescued by **local** (either-side) context; hard/semantic tokens need **specific/far** tokens (high variance). **Right-only context still helping is a clean diffusion-specific result** (an AR model could not use it) — highlight it.
**Pitfalls.** Hold revealed-token **count** constant across conditions; "far random" only helps when it happens to include the relevant token → average a lot. (Optional oracle: reveal top-k tokens by attention/gradient relevance to `i` for a best-case upper bound.)

### Exp 4 — Rescue & false rescue, by difficulty (harness H2, real validation data — the only experiment using gold targets)
**Setup.** On **real** sequences (gold known), keep `i` masked across the full ladder `r ∈ R`; record `H_i(r)`, `NLL_i(r)`, and `A_i` at the late ratio. Group by `D_i`. This is where the entropy story (exp 2/3) is cross-cut with **correctness** — the distinct contribution of exp 4.
**Metric.** Both rescue scores `Rescue^H_i`, `Rescue^{NLL}_i`; **false-rescue** flag; the **2×2×2 taxonomy** (hard/easy × rescued/not × correct/wrong) — report mass per cell + which linguistic categories populate the interesting cells (hard+rescued+correct = successful resolution; hard+persistent+wrong = persistent ambiguity; **false rescue** = entropy down but wrong; easy+wrong = surprising failure).
**Predicted.** A meaningful fraction of hard tokens are rescuable; some show false rescue (confidently wrong). 
**Note (mechanistic link, observational only):** in *actual* generation a rescuable token can be **frozen early** (random τ) before its context arrives → "potential rescue" (measured here in H2) vs "realized rescue" gap. We only *observe* this gap (no sampler change); it is the motivation for later difficulty-aware decoding (§6).
**Pitfalls.** Open-text accuracy undercounts → lean on gold-NLL + top-k.

### Exp 5 — Are initially-hard tokens more important? Perturb-and-continue (harness H1)
**Setup.** Run the sampler to ratio `r ∈ {20,40,60,80}%` (≥ some tokens committed; `r=0` excluded — nothing to perturb). Among **currently-committed** positions, rank by `D_i` (initial entropy). **Replace a 20% subset's committed token with a wrong token** (a uniformly random different vocab id; alt: the least-likely token under `p_x0`), then **continue the unmodified sampler to completion**. Three conditions: **top-20% hardest**, **random-20%**, **bottom-20% easiest**.
**Metric.** Final-output quality drop vs unperturbed: **generative perplexity** (GPT-2-large, already in [`metrics.py`](metrics.py#L161)); optionally change in NLL over the *unperturbed* positions.
**Predicted (competing hypotheses — state both).** (i) 80/20 view: high-difficulty tokens are pivotal "forks" → largest quality drop. (ii) Counter-view: high-entropy positions are "free slots" (many valid fillers) → *smaller* drop. Which dominates tests whether *difficulty* implies *importance* in diffusion.
**Pitfalls.** Control **spatial clustering** (hard tokens may bunch; random spreads) — match spatial distribution or normalize per token. Define "wrong token" consistently. Run enough sequences for stable gen-PPL.

---

## 5. Implementation — **DONE** (run guide & file map: [README_experiments.md](README_experiments.md))

Built as the [`denoise_diff/`](denoise_diff/) package that **calls the existing model methods** (`_continuous_step`, `_discrete_step`, `forward`, `prior_sample`) — `algo.py` and the model are untouched (only `trainer_base.py`'s `hydra.utils` import was made lazy).

- [`denoise_diff/metrics.py`](denoise_diff/metrics.py) — predictive entropy, gold-NLL, top-k accuracy, quantile buckets, reveal-set builders (near/far/left/right/random).
- [`denoise_diff/harness.py`](denoise_diff/harness.py) — `trace_sample` (exp 1/2), `denoiser_probe` (exp 3/4), `gold_reconstruct_trace` (exp 6), `gold_renoise_trace` (exp 7).
- [`denoise_diff/model.py`](denoise_diff/model.py) — `load_model` (from the ckpt config, omegaconf-only), `load_gold`, `gen_ppl`.
- [`denoise_diff/plotting.py`](denoise_diff/plotting.py) — shared figure helpers.
- [`run_experiments.py`](run_experiments.py) — CLI to collect data → `experiments/out/<run>/*.pt`.
- [`analyze.py`](analyze.py) — CLI to make figures (views: standard / per_ratio / difftime / exp6 / exp7).

Exp 5 (out-of-distribution perturbation) was retired in favour of exp 7 (in-distribution re-noise & resample). See the README for the exp↔code↔figure map and the run commands.

---

## 6. Risks, accepted limitations, future
- **Checkpoint available** (§0); remaining gotchas are matching trained algo knobs (`step_size=0.5`, `sigma_min/max=0.1/4.0`) and the `_orig_mod`/EMA loading details.
- **Accepted confound (§1.2):** entropy drops mix neighbor-context with the masked token's own continuous-noise reduction; not decoupled this run. *Future control:* hold target `i`'s continuous σ fixed while varying others' discrete reveal.
- **Entropy timing (§1.3):** measure pre-commit or get degenerate zeros.
- **Open-text accuracy** undercounts (lexical variation) → gold-NLL + top-k; `text8` for crisp accuracy.
- **Special tokens / `ignore_bos`** — filter pinned positions in all analyses.
- **Out of scope for now (future):** modifying the reveal rule to **difficulty-aware / confidence-ordered decoding** to test the causal 80/20 question and to convert exp-4's "potential vs realized rescue" gap into an intervention; and Difficulty B (angle concentration).

---

## 7. Execution checklist
1. [ ] Load the ckpt (`model=small`; override `step_size`/`sigma_*`; handle `_orig_mod`/EMA); sanity-check `mode=sample_eval` output.
2. [ ] Implement H1 trajectory logging + `difficulty.py` (entropy; gold-NLL for exp 4); validate on a few sequences.
3. [ ] **Exp 1** (H1) — confirm order ⊥ difficulty. **Exp 2** (H1, reference-free) — entropy-collapse curves.
4. [ ] **Exp 3** (H2 structured probe, reference-free) near/far + left/right; **Exp 4** (H2, real data) rescue/false-rescue taxonomy. Ratios `{0,20,40,60,80}`.
5. [ ] **Exp 5** perturb-and-continue (replace-with-wrong), ratios `{20,40,60,80}`, 3 conditions, gen-PPL.
6. [ ] Write up: per-experiment plots + the order⊥difficulty / collapse / rescue / importance narrative.
