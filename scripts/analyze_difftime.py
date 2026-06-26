#!/usr/bin/env python
"""Difficulty-at-X% analysis: bucket easy/hard by entropy measured at 10/20/30/40% denoised
(instead of 0%), separately for each reference ratio. Reads the dumps from ``--in`` and writes
``*_difftime_diff{R}.png`` to ``--figdir``.

Difficulty score D_i(rho) per experiment's own regime:
  exp1/2 : entropy along the sampler trajectory at the step where rho% are denoised (self-gen).
  exp3   : target entropy under rho% RANDOM context (saved as diff_r{R}).
  exp4   : target entropy under rho% GOLD context (= entropy at that ratio index).
Only tokens still masked at rho are bucketed.
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
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, name)
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


def _grouped_bars(ax, xlabels, series, ylabel, title):
    n = len(series); x = np.arange(len(xlabels)); w = 0.8 / max(n, 1)
    for i, (lbl, ys) in enumerate(series.items()):
        ax.bar(x + (i - (n - 1) / 2) * w, ys, w, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set(ylabel=ylabel, title=title); ax.legend(fontsize=8)


# --------------------------------------------------------------------------- #
def exp1_2(indir, figdir, rho):
    f = os.path.join(indir, "exp1_2_trace.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    H = d["H_traj"].float(); rs = d["reveal_step"].long()
    S, B, L = H.shape
    rps = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
    s0 = int(torch.argmin((rps - rho).abs()))                 # step nearest rho
    masked = rs > s0                                          # tokens still masked at rho
    if masked.sum() < 20:
        return
    Dvals = H[s0][masked]                                     # D_i(rho)
    gq = D.quantile_groups(Dvals, 4)
    groups = torch.full((B, L), -1, dtype=torch.long); groups[masked] = gq

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    # left: collapse from rho onward, by D_i(rho) quartile
    xs = rps[s0:].numpy()
    for g in range(4):
        ys = [H[s][(rs > s) & (groups == g)].mean().item() if ((rs > s) & (groups == g)).sum() > 0
              else np.nan for s in range(s0, S)]
        ax[0].plot(xs, ys, label=f"q{g}" + ("(easy)" if g == 0 else "(hard)" if g == 3 else ""))
    ax[0].set(title=f"Exp2 | difficulty @ {int(rho*100)}% denoised\nentropy collapse of tokens masked at {int(rho*100)}%",
              xlabel="reveal ratio", ylabel="mean entropy (nats)"); ax[0].legend(fontsize=8)
    # right: null control -- does reveal order after rho depend on D_i(rho)?
    later = rs[masked].float() / max(S - 1, 1)
    sp = spearmanr(Dvals.numpy(), later.numpy()).statistic if spearmanr else float("nan")
    ax[1].hexbin(later.numpy(), Dvals.numpy(), gridsize=35, cmap="magma", mincnt=1)
    ax[1].set(title=f"Exp1 null control @ {int(rho*100)}%: reveal order vs D_i(rho)\nSpearman={sp:.3f} (expect ~0)",
              xlabel="reveal fraction (of tokens masked at rho)", ylabel=f"entropy at {int(rho*100)}% (nats)")
    _save(fig, figdir, f"exp1_2_difftime_diff{int(rho*100)}.png")


# --------------------------------------------------------------------------- #
def exp3(indir, figdir, rho):
    f = os.path.join(indir, "exp3_structured.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    key = f"diff_r{int(rho*100)}"
    if key not in d:
        return
    ratios, modes = d["ratios"], d["modes"]
    hard = D.quantile_groups(d[key].float(), 2) == 1
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.2 * len(ratios), 4.5), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), ratios):
        series = {lbl: [d[f"ent_r{int(r*100)}_{m}"].float()[msk].mean().item() for m in modes]
                  for lbl, msk in [("easy", ~hard), ("hard", hard)]}
        _grouped_bars(ax, modes, series, "entropy at target (nats)", f"{int(r*100)}% revealed")
    fig.suptitle(f"Exp3 | difficulty @ {int(rho*100)}% denoised: context type by easy/hard")
    _save(fig, figdir, f"exp3_difftime_diff{int(rho*100)}.png")


# --------------------------------------------------------------------------- #
def exp4(indir, figdir, rho):
    f = os.path.join(indir, "exp4_rescue.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    ratios = d["ratios"]; tm = d["target_mask"].bool(); R = d["entropy"].shape[0]
    iro = int(np.argmin([abs(r - rho) for r in ratios]))      # ratio index nearest rho
    sel = tm.unsqueeze(0).expand(R, -1, -1)
    Ht = d["entropy"].float()[sel].reshape(R, -1)
    Nt = d["gold_nll"].float()[sel].reshape(R, -1)
    At = d["correct"].float()[sel].reshape(R, -1)
    hard = D.quantile_groups(Ht[iro], 2) == 1                  # difficulty at rho (gold)
    pcts = [f"{int(r*100)}%" for r in ratios]
    panels = [(Ht, "entropy (nats)", "entropy"), (Nt, "gold NLL (nats)", "gold-NLL"),
              (At, "accuracy", "top-1 accuracy")]
    if "top3" in d:
        panels.append((d["top3"].float()[sel].reshape(R, -1), "accuracy", "top-3 accuracy"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.5))
    for ax, (M, ylab, ttl) in zip(np.atleast_1d(axes), panels):
        series = {"easy": [M[i][~hard].mean().item() for i in range(R)],
                  "hard": [M[i][hard].mean().item() for i in range(R)]}
        _grouped_bars(ax, pcts, series, ylab, f"{ttl} (difficulty @ {int(rho*100)}%)")
    fig.suptitle(f"Exp4 | difficulty bucketed by entropy at {int(rho*100)}% gold context")
    _save(fig, figdir, f"exp4_difftime_diff{int(rho*100)}.png")


def exp4_forward(indir, figdir, rho):
    """Exp4 as forward-only CURVES from the bucketing ratio rho (entropy/NLL/top1/top3)."""
    f = os.path.join(indir, "exp4_rescue.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    ratios = d["ratios"]; tm = d["target_mask"].bool(); R = d["entropy"].shape[0]
    iro = int(np.argmin([abs(r - rho) for r in ratios]))
    sel = tm.unsqueeze(0).expand(R, -1, -1)
    Ht = d["entropy"].float()[sel].reshape(R, -1)
    Nt = d["gold_nll"].float()[sel].reshape(R, -1)
    At = d["correct"].float()[sel].reshape(R, -1)
    hard = D.quantile_groups(Ht[iro], 2) == 1
    fwd = [i for i, r in enumerate(ratios) if r >= rho - 1e-6]      # forward from rho only
    xs = [ratios[i] for i in fwd]
    panels = [(Ht, "entropy (nats)", "entropy"), (Nt, "gold NLL (nats)", "gold-NLL"),
              (At, "accuracy", "top-1 accuracy")]
    if "top3" in d:
        panels.append((d["top3"].float()[sel].reshape(R, -1), "accuracy", "top-3 accuracy"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3))
    for ax, (M, ylab, ttl) in zip(np.atleast_1d(axes), panels):
        ax.plot(xs, [M[i][~hard].mean().item() for i in fwd], marker="o", label="easy")
        ax.plot(xs, [M[i][hard].mean().item() for i in fwd], marker="o", label="hard")
        ax.axvline(rho, color="k", ls=":", lw=0.8)
        ax.set(xlabel="reveal ratio", ylabel=ylab, title=ttl); ax.legend(fontsize=8)
    fig.suptitle(f"Exp4 forward from {int(rho*100)}%: easy vs hard (bucketed at {int(rho*100)}% gold context)")
    _save(fig, figdir, f"exp4_difftime_fwd_diff{int(rho*100)}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="experiments/out/run_difftime")
    ap.add_argument("--figdir", default="gen_imgs/denoise_difftime")
    ap.add_argument("--diff-ratios", default="10,20,30,40")
    a = ap.parse_args()
    rhos = [int(x) / 100.0 for x in a.diff_ratios.split(",")]
    for rho in rhos:
        print(f"=== difficulty reference {int(rho*100)}% ===")
        exp1_2(a.indir, a.figdir, rho)
        exp3(a.indir, a.figdir, rho)
        exp4(a.indir, a.figdir, rho)
        exp4_forward(a.indir, a.figdir, rho)
    print("done.")


if __name__ == "__main__":
    main()
