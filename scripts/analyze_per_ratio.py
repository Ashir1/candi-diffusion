#!/usr/bin/env python
"""Per-start-denoise-ratio views of experiments 1-4 (clearer than the overlaid figures).

For each fixed reveal ratio in {0,20,40,60,80}% it draws grouped bars so you can read
values off at each denoise percentage. Writes ``*_per_ratio.png`` to ``--figdir``.
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

RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8]
PCTS = [f"{int(r*100)}%" for r in RATIOS]


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, name)
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


def _grouped_bars(ax, group_labels, series, ylabel, title, annotate=False):
    """series: dict label -> list(len=len(group_labels)). Draws grouped bars."""
    n = len(series); x = np.arange(len(group_labels)); w = 0.8 / max(n, 1)
    for i, (lbl, ys) in enumerate(series.items()):
        bars = ax.bar(x + (i - (n - 1) / 2) * w, ys, w, label=lbl)
        if annotate:
            for b, y in zip(bars, ys):
                if np.isfinite(y):
                    ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}",
                            ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(group_labels)
    ax.set(ylabel=ylabel, title=title); ax.legend(fontsize=8)


# --------------------------------------------------------------------------- #
def exp1_2(indir, figdir):
    f = os.path.join(indir, "exp1_2_trace.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    H = d["H_traj"].float(); rs = d["reveal_step"].long(); H0 = d["H0"].float()
    S, B, L = H.shape
    ratio_per_step = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
    star = [int(torch.argmin((ratio_per_step - r).abs()).item()) for r in RATIOS]  # nearest step per ratio
    groups = D.quantile_groups(H0, 4).reshape(B, L)

    # --- Exp1 per ratio: committed-token entropy vs still-masked cohort, at each ratio ---
    comm, coh = [], []
    for i in star:
        c = (rs == i); m = (rs > i)
        comm.append(H[i][c].mean().item() if c.sum() > 0 else np.nan)
        coh.append(H[i][m].mean().item() if m.sum() > 0 else np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _grouped_bars(ax, PCTS, {"just-committed": comm, "still-masked cohort": coh},
                  "mean entropy (nats)",
                  "Exp1 per ratio: committed vs masked-cohort entropy\n(equal at every ratio => reveal order independent of difficulty)",
                  annotate=True)
    _save(fig, figdir, "exp1_per_ratio.png")

    # --- Exp2 per ratio: still-masked entropy by initial-difficulty quartile ---
    series = {}
    for g in range(4):
        ys = []
        for i in star:
            msk = (rs > i) & (groups == g)
            ys.append(H[i][msk].mean().item() if msk.sum() > 0 else np.nan)
        series[f"D_i q{g}" + ("(easy)" if g == 0 else "(hard)" if g == 3 else "")] = ys
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _grouped_bars(ax, PCTS, series, "mean entropy (nats)",
                  "Exp2 per ratio: entropy of still-masked tokens by initial difficulty")
    _save(fig, figdir, "exp2_per_ratio.png")


# --------------------------------------------------------------------------- #
def exp3(indir, figdir):
    f = os.path.join(indir, "exp3_structured.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    ratios, modes = d["ratios"], d["modes"]
    Di = d[f"ent_r{int(ratios[0]*100)}_{modes[0]}"].float()
    hard = D.quantile_groups(Di, 2) == 1
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.2 * len(ratios), 4.5), sharey=True)
    for ax, r in zip(np.atleast_1d(axes), ratios):
        series = {}
        for lbl, msk in [("easy", ~hard), ("hard", hard)]:
            series[lbl] = [d[f"ent_r{int(r*100)}_{m}"].float()[msk].mean().item() for m in modes]
        _grouped_bars(ax, modes, series, "entropy at target (nats)", f"{int(r*100)}% revealed")
    fig.suptitle("Exp3 per ratio: which context (near/far/left/right) collapses entropy, easy vs hard")
    _save(fig, figdir, "exp3_per_ratio.png")


# --------------------------------------------------------------------------- #
def exp4(indir, figdir):
    f = os.path.join(indir, "exp4_rescue.pt")
    if not os.path.exists(f):
        return
    d = torch.load(f, map_location="cpu", weights_only=False)
    ratios = d["ratios"]; tm = d["target_mask"].bool()
    R = d["entropy"].shape[0]
    sel = tm.unsqueeze(0).expand(R, -1, -1)
    Ht = d["entropy"].float()[sel].reshape(R, -1)
    Nt = d["gold_nll"].float()[sel].reshape(R, -1)
    At = d["correct"].float()[sel].reshape(R, -1)
    hard = D.quantile_groups(Ht[0], 2) == 1
    pcts = [f"{int(r*100)}%" for r in ratios]

    panels = [(Ht, "entropy (nats)", "entropy"), (Nt, "gold NLL (nats)", "gold-NLL"),
              (At, "accuracy", "top-1 accuracy")]
    if "top3" in d:
        panels.append((d["top3"].float()[sel].reshape(R, -1), "accuracy", "top-3 accuracy"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.5))
    for ax, (M, ylab, title) in zip(np.atleast_1d(axes), panels):
        series = {"easy": [M[i][~hard].mean().item() for i in range(R)],
                  "hard": [M[i][hard].mean().item() for i in range(R)]}
        _grouped_bars(ax, pcts, series, ylab, f"Exp4 per ratio: {title} (kept-masked targets)")
    _save(fig, figdir, "exp4_per_ratio.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="experiments/out/run1")
    ap.add_argument("--figdir", default="gen_imgs/denoise")
    a = ap.parse_args()
    exp1_2(a.indir, a.figdir); exp3(a.indir, a.figdir); exp4(a.indir, a.figdir)
    print("done.")


if __name__ == "__main__":
    main()
