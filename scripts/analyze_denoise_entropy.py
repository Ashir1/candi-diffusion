#!/usr/bin/env python
"""Analysis + figures for the denoise-vs-difficulty experiment suite.

Reads the tensors dumped by ``run_denoise_entropy_experiment.py`` from ``--in`` and
writes plots to ``--figdir`` (default gen_imgs/denoise/). Each experiment is guarded
by file existence, so partial runs still plot.
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import difficulty as D  # noqa: E402

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = pearsonr = None

FORK_CUT = 0.672  # 2506.01939 forking-entropy cutoff (nats)


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, name)
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    sp = spearmanr(x[m], y[m]).statistic if spearmanr else float("nan")
    pr = pearsonr(x[m], y[m])[0] if pearsonr else float("nan")
    return sp, pr


# --------------------------------------------------------------------------- #
def analyze_exp12(indir, figdir):
    f = os.path.join(indir, "exp1_2_trace.pt")
    if not os.path.exists(f):
        return
    print("[exp1/2] analyzing", f)
    d = torch.load(f, map_location="cpu")
    H_traj = d["H_traj"].float()                     # (S,B,L)
    rs = d["reveal_step"].long()                     # (B,L)
    H0 = d["H0"].float()                             # (B,L)
    S, B, L = H_traj.shape
    rs_c = rs.clamp(0, S - 1)

    # entropy at reveal: H_traj[reveal_step, b, l]
    bb = torch.arange(B)[:, None].expand(B, L)
    ll = torch.arange(L)[None, :].expand(B, L)
    H_at = H_traj[rs_c, bb, ll]                       # (B,L)
    reveal_frac = rs.float() / max(S - 1, 1)

    # ---- Exp 1: committed vs still-masked cohort, per step ----
    steps, diff = [], []
    for i in range(S):
        committed = (rs == i)
        masked_before = (rs >= i)
        if committed.sum() > 0 and masked_before.sum() > 0:
            steps.append(i / max(S - 1, 1))
            diff.append((H_traj[i][committed].mean() - H_traj[i][masked_before].mean()).item())

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    sp, pr = _corr(reveal_frac.flatten(), H_at.flatten())
    ax[0].hexbin(reveal_frac.flatten(), H_at.flatten(), gridsize=40, cmap="viridis", mincnt=1)
    ax[0].set(title=f"Exp1/2: reveal frac vs entropy-at-reveal\nSpearman={sp:.3f} Pearson={pr:.3f}",
              xlabel="reveal fraction (tau/S)", ylabel="entropy at reveal (nats)")
    sp0, pr0 = _corr(reveal_frac.flatten(), H0.flatten())
    ax[1].hexbin(reveal_frac.flatten(), H0.flatten(), gridsize=40, cmap="magma", mincnt=1)
    ax[1].set(title=f"NULL CONTROL: reveal frac vs initial difficulty H0\nSpearman={sp0:.3f} (expect ~0)",
              xlabel="reveal fraction", ylabel="H0 (nats)")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].plot(steps, diff, ".", ms=3, alpha=0.6)
    ax[2].set(title="Exp1: committed - masked-cohort entropy per step\n(expect ~0 => order independent of difficulty)",
              xlabel="reveal fraction", ylabel="mean entropy diff (nats)")
    _save(fig, figdir, "exp1_order_vs_difficulty.png")

    # ---- Exp 2: entropy-collapse curves by initial-difficulty quartile ----
    groups = D.quantile_groups(H0, 4).reshape(B, L)
    ratio_per_step = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
    fig, ax = plt.subplots(figsize=(7, 5))
    for g in range(4):
        ys = []
        for i in range(S):
            masked = (rs > i) & (groups == g)
            ys.append(H_traj[i][masked].mean().item() if masked.sum() > 0 else np.nan)
        ax.plot(ratio_per_step.numpy(), ys, label=f"D_i quartile {g} ({'easy' if g==0 else 'hard' if g==3 else ''})")
    ax.set(title="Exp2: entropy collapse of still-masked tokens vs reveal ratio",
           xlabel="global reveal ratio", ylabel="mean entropy (nats)")
    ax.legend()
    _save(fig, figdir, "exp2_entropy_collapse.png")


# --------------------------------------------------------------------------- #
def analyze_exp3(indir, figdir):
    f = os.path.join(indir, "exp3_structured.pt")
    if not os.path.exists(f):
        return
    print("[exp3] analyzing", f)
    d = torch.load(f, map_location="cpu")
    ratios, modes = d["ratios"], d["modes"]
    r0 = int(ratios[0] * 100)
    Di = d[f"ent_r{r0}_{modes[0]}"].float()           # no-context entropy = initial difficulty
    groups = D.quantile_groups(Di, 4)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
    for g in range(4):
        ax = axes[g]
        for mode in modes:
            ys = [d[f"ent_r{int(r*100)}_{mode}"].float()[groups == g].mean().item() for r in ratios]
            ax.plot([r for r in ratios], ys, marker="o", label=mode)
        ax.set(title=f"D_i quartile {g}", xlabel="reveal ratio")
        if g == 0:
            ax.set_ylabel("mean entropy at target (nats)")
        ax.legend(fontsize=8)
    fig.suptitle("Exp3: which context collapses entropy (near/far/left/right), by difficulty")
    _save(fig, figdir, "exp3_structured_context.png")


# --------------------------------------------------------------------------- #
def analyze_exp4(indir, figdir):
    f = os.path.join(indir, "exp4_rescue.pt")
    if not os.path.exists(f):
        return
    print("[exp4] analyzing", f)
    d = torch.load(f, map_location="cpu")
    ratios = d["ratios"]
    tm = d["target_mask"].bool()                      # (B,L)
    Hr = d["entropy"].float()                         # (R,B,L)
    Nr = d["gold_nll"].float()
    Cr = d["correct"].float()
    R = Hr.shape[0]
    sel = tm.unsqueeze(0).expand(R, -1, -1)
    H = Hr[sel].reshape(R, -1)                         # (R, n_targets)
    N = Nr[sel].reshape(R, -1)
    Cacc = Cr[sel].reshape(R, -1)

    Di = H[0]                                          # initial entropy at r0
    groups = D.quantile_groups(Di, 2)                 # 0=easy,1=hard
    hard = groups == 1

    # collapse curves (entropy + gold-NLL), hard vs easy
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for lbl, msk in [("easy", ~hard), ("hard", hard)]:
        ax[0].plot(ratios, [H[i][msk].mean().item() for i in range(R)], marker="o", label=lbl)
        ax[1].plot(ratios, [N[i][msk].mean().item() for i in range(R)], marker="o", label=lbl)
    ax[0].set(title="Exp4: entropy collapse (kept-masked targets)", xlabel="reveal ratio", ylabel="entropy (nats)")
    ax[1].set(title="Exp4: gold-NLL collapse", xlabel="reveal ratio", ylabel="gold NLL (nats)")
    ax[0].legend(); ax[1].legend()
    _save(fig, figdir, "exp4_rescue_curves.png")

    # rescue scatter + false rescue, and taxonomy
    rescue_H = (H[0] - H[-1])
    rescue_N = (N[0] - N[-1])
    nll_cut = torch.quantile(N[-1], 0.75).item()
    false_rescue = (H[-1] < FORK_CUT) & (N[-1] > nll_cut)
    correct_final = Cacc[-1] > 0.5
    rescued = rescue_N > 0

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
    ax[0].scatter(rescue_H[~false_rescue], rescue_N[~false_rescue], s=6, alpha=0.4, label="normal")
    ax[0].scatter(rescue_H[false_rescue], rescue_N[false_rescue], s=10, alpha=0.7, color="red", label="false rescue")
    ax[0].set(title="Exp4: entropy-rescue vs gold-NLL-rescue\n(false rescue = confident but wrong)",
              xlabel="entropy rescue  H0-Hlate", ylabel="gold-NLL rescue  NLL0-NLLlate")
    ax[0].legend()

    # 2x2x2 taxonomy masses
    cells, masses = [], []
    for hl, hm in [("hard", hard), ("easy", ~hard)]:
        for rl, rmsk in [("rescued", rescued), ("not", ~rescued)]:
            for cl, cmsk in [("correct", correct_final), ("wrong", ~correct_final)]:
                cells.append(f"{hl}\n{rl}\n{cl}")
                masses.append(float((hm & rmsk & cmsk).float().mean().item()))
    ax[1].bar(range(len(cells)), masses)
    ax[1].set_xticks(range(len(cells))); ax[1].set_xticklabels(cells, fontsize=6)
    ax[1].set(title="Exp4: hard/easy x rescued/not x correct/wrong (fraction of targets)",
              ylabel="fraction")
    _save(fig, figdir, "exp4_rescue_taxonomy.png")
    print(f"  false-rescue fraction: {false_rescue.float().mean().item():.3f}")


# --------------------------------------------------------------------------- #
def analyze_exp5(indir, figdir):
    f = os.path.join(indir, "exp5_perturb.pt")
    if not os.path.exists(f):
        return
    print("[exp5] analyzing", f)
    d = torch.load(f, map_location="cpu")
    runs = d["runs"]
    # index baselines by ratio
    base = {e["ratio"]: e["final_tokens"] for e in runs if e["selection"] == "None"}
    ratios = sorted(base.keys())
    sels = ["hard", "random", "easy"]
    drift = {s: [] for s in sels}
    ppl = {s: [] for s in sels}
    for r in ratios:
        for s in sels:
            e = next(x for x in runs if x["ratio"] == r and x["selection"] == s)
            keep = ~e["perturb_mask"]                  # downstream (non-perturbed) positions
            changed = (e["final_tokens"] != base[r]) & keep
            drift[s].append(changed.float().sum().item() / keep.float().sum().item())
            ppl[s].append(e.get("gen_ppl", float("nan")))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(ratios)); w = 0.25
    for i, s in enumerate(sels):
        ax[0].bar(x + (i - 1) * w, drift[s], w, label=s)
    ax[0].set(title="Exp5: downstream drift vs baseline\n(fraction of non-perturbed tokens changed)",
              xlabel="perturb ratio", ylabel="drift")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{r:.0%}" for r in ratios]); ax[0].legend()
    if any(np.isfinite(ppl["hard"])):
        for i, s in enumerate(sels):
            ax[1].bar(x + (i - 1) * w, ppl[s], w, label=s)
        ax[1].set(title="Exp5: generative perplexity by condition", xlabel="perturb ratio", ylabel="gen PPL")
        ax[1].set_xticks(x); ax[1].set_xticklabels([f"{r:.0%}" for r in ratios]); ax[1].legend()
    else:
        ax[1].text(0.5, 0.5, "gen_ppl not computed\n(rerun with --gen-ppl)", ha="center")
    _save(fig, figdir, "exp5_perturbation_importance.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="experiments/out/run1")
    ap.add_argument("--figdir", default="gen_imgs/denoise")
    args = ap.parse_args()
    analyze_exp12(args.indir, args.figdir)
    analyze_exp3(args.indir, args.figdir)
    analyze_exp4(args.indir, args.figdir)
    analyze_exp5(args.indir, args.figdir)
    print("done.")


if __name__ == "__main__":
    main()
