"""Shared figure helpers for the analysis views in ../analyze.py."""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from scipy.stats import spearmanr, pearsonr
except Exception:
    spearmanr = pearsonr = None

FORK_CUT = 0.672                                       # 2506.01939 forking-entropy cutoff (nats)
COLORS = {"hard": "C3", "random": "C1", "easy": "C0", "baseline": "gray"}   # consistent convention


def load(indir, name):
    """torch.load ``indir/name`` (or None if missing -- callers guard partial runs)."""
    p = os.path.join(indir, name)
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else None


def save_fig(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, name)
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


def np_(t):
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


def grouped_bars(ax, xlabels, series, ylabel, title, annotate=False):
    """series: dict label -> list (len == len(xlabels)). hard/random/easy/baseline get fixed colours."""
    n = len(series); x = np.arange(len(xlabels)); w = 0.8 / max(n, 1)
    for i, (lbl, ys) in enumerate(series.items()):
        key = lbl.split()[0] if isinstance(lbl, str) else lbl
        bars = ax.bar(x + (i - (n - 1) / 2) * w, ys, w, label=lbl, color=COLORS.get(key))
        if annotate:
            for b, y in zip(bars, ys):
                if np.isfinite(y):
                    ax.text(b.get_x() + b.get_width() / 2, y, f"{y:.2f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set(ylabel=ylabel, title=title); ax.legend(fontsize=8)


def corr(x, y):
    """(spearman, pearson), ignoring nan; (nan, nan) if too few points."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan")
    sp = spearmanr(x[m], y[m]).statistic if spearmanr else float("nan")
    pr = pearsonr(x[m], y[m])[0] if pearsonr else float("nan")
    return sp, pr
