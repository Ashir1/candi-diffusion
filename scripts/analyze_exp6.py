#!/usr/bin/env python
"""Exp6 analysis: rescue under the sampler's stochastic reveal schedule (real text).

For each bucketing ratio rho in {10,20,30,40}% it buckets the tokens still masked at rho by
their entropy there, then for each token uses metrics at its OWN reveal time tau (where it
accumulated the most context). Produces, per rho:
  exp6_rescue_diff{R}.png      -- entropy-rescue vs gold-NLL-rescue scatter (false rescue red)
                                  + the hard/easy x rescued/not x correct/wrong taxonomy (top-5).
  exp6_time_vs_rescue_diff{R}.png -- "more time -> rescue": fraction correct (top-3, top-5) vs how
                                     much context the token had at reveal, for hard vs easy.
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

FORK_CUT = 0.672


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, name)
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


def analyze(indir, figdir, rho):
    f = os.path.join(indir, "exp6_recon.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    H = d["H_traj"].float(); rs = d["reveal_step"].long()
    nll0 = d["nll0"].float(); nllr = d["nll_reveal"].float()
    t3 = d["top3_reveal"].float(); t5 = d["top5_reveal"].float()
    S, N, L = H.shape
    rps = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
    s0 = int(torch.argmin((rps - rho).abs()))
    masked = rs > s0                                          # tokens not yet committed at rho
    if masked.sum() < 30:
        return

    # entropy at reveal time (gathered), and at step 0
    rsc = rs.clamp(0, S - 1)
    nn = torch.arange(N)[:, None].expand(N, L)
    ll = torch.arange(L)[None, :].expand(N, L)
    H_rev = H[rsc, nn, ll]                                    # (N,L)
    H0 = H[0]

    m = masked
    Dv = H[s0][m]
    hard = Dv > Dv.median()
    rescue_H = (H0 - H_rev)[m]
    rescue_N = (nll0 - nllr)[m]
    t3m, t5m = t3[m], t5[m]
    H_revm = H_rev[m]
    rescued = rescue_N > 0
    correct = t5m > 0.5                                       # lenient: top-5
    false_rescue = (H_revm < FORK_CUT) & (t5m < 0.5)
    ctx_at_reveal = rps[rs[m]]                                # fraction revealed when this token committed

    # ---- figure 1: scatter + taxonomy ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
    ax[0].scatter(rescue_H[~false_rescue], rescue_N[~false_rescue], s=6, alpha=0.35, label="normal")
    ax[0].scatter(rescue_H[false_rescue], rescue_N[false_rescue], s=12, color="red", alpha=0.8, label="false rescue")
    ax[0].set(title=f"Exp6 @ {int(rho*100)}%: entropy-rescue vs gold-NLL-rescue\n(false rescue = confident but not top-5)",
              xlabel="entropy rescue  H0 - H(reveal)", ylabel="gold-NLL rescue  NLL0 - NLL(reveal)")
    ax[0].legend()
    cells, mass = [], []
    for hl, hm in [("hard", hard), ("easy", ~hard)]:
        for rl, rm in [("rescued", rescued), ("not", ~rescued)]:
            for cl, cm in [("correct", correct), ("wrong", ~correct)]:
                cells.append(f"{hl}\n{rl}\n{cl}"); mass.append(float((hm & rm & cm).float().mean()))
    ax[1].bar(range(8), mass)
    ax[1].set_xticks(range(8)); ax[1].set_xticklabels(cells, fontsize=6)
    ax[1].set(title=f"Exp6 @ {int(rho*100)}%: hard/easy x rescued/not x correct/wrong (top-5)", ylabel="fraction")
    _save(fig, figdir, f"exp6_rescue_diff{int(rho*100)}.png")

    # ---- figure 2: more time -> rescue ----
    fig, ax = plt.subplots(figsize=(7.5, 5))
    edges = np.linspace(float(ctx_at_reveal.min()), float(ctx_at_reveal.max()), 7)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for grp, gm, col in [("hard", hard, "C3"), ("easy", ~hard, "C0")]:   # color = group
        c = ctx_at_reveal[gm].numpy()
        for acc, lbl, ls in [(t5m[gm].numpy(), "top-5", "-"), (t3m[gm].numpy(), "top-3", "--")]:  # style = metric
            ys = [acc[(c >= edges[i]) & (c < edges[i + 1] + (i == len(edges) - 2) * 1e-6)].mean()
                  if ((c >= edges[i]) & (c < edges[i + 1] + 1e-6)).sum() > 0 else np.nan
                  for i in range(len(edges) - 1)]
            ax.plot(centers, ys, ls, color=col, marker="o", label=f"{grp} {lbl}")
    ax.set(title=f"Exp6 @ {int(rho*100)}%: more reveal-time (context) -> rescue\n(tokens still masked at {int(rho*100)}%)",
           xlabel="context revealed when token committed (reveal time)", ylabel="fraction correct vs gold")
    ax.legend(fontsize=8)
    _save(fig, figdir, f"exp6_time_vs_rescue_diff{int(rho*100)}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="experiments/out/run_big")
    ap.add_argument("--figdir", default="gen_imgs/denoise_big_difftime")
    ap.add_argument("--diff-ratios", default="10,20,30,40")
    a = ap.parse_args()
    for rho in [int(x) / 100.0 for x in a.diff_ratios.split(",")]:
        print(f"=== exp6 difficulty reference {int(rho*100)}% ===")
        analyze(a.indir, a.figdir, rho)
    print("done.")


if __name__ == "__main__":
    main()
