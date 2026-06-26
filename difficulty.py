"""Token-difficulty metrics and reveal-set builders for the denoise-vs-difficulty study.

All functions are pure (no model state) so that "swap in another notion of
difficulty" lives in this one file.

Conventions
-----------
* ``p``     : predicted clean distribution over the real vocab, shape (..., V) -- CANDI's ``p_x0``/``denoised``.
* ``logp``  : log of the above (what ``model.forward`` returns).
* Entropy is reported in **nats** (natural log), matching 2506.01939 Eq.1
  (``H = -sum_v p_v log p_v``; their high-entropy "forking" cutoff ~= 0.672 nats).
* ``gold``  : ground-truth token ids, shape (...,) -- only available for exp 4.
"""

import math

import torch

EPS = 1e-12
LOG2 = math.log(2.0)


# --------------------------------------------------------------------------- #
# Difficulty A: predictive entropy (2506.01939, Eq. 1)
# --------------------------------------------------------------------------- #
def predictive_entropy(p: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (nats) of a probability tensor along the last dim -> (...,)."""
    p = p.clamp_min(0)
    logp = p.clamp_min(EPS).log()
    return -(p * logp).sum(dim=-1)


def entropy_from_logprobs(logp: torch.Tensor) -> torch.Tensor:
    """Same entropy but from log-probabilities (numerically nicer) -> (...,)."""
    return -(logp.exp() * logp).sum(dim=-1)


def nats_to_bits(h: torch.Tensor) -> torch.Tensor:
    return h / LOG2


# --------------------------------------------------------------------------- #
# Correctness metrics -- EXPERIMENT 4 ONLY (need ground-truth ``gold`` tokens)
# --------------------------------------------------------------------------- #
def gold_nll(logp: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """-log p(gold) per position -> (...,). ``logp`` is (...,V), ``gold`` is (...,)."""
    return -logp.gather(dim=-1, index=gold.unsqueeze(-1)).squeeze(-1)


def argmax_accuracy(logp_or_p: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """1 where argmax == gold -> (...,) float."""
    return (logp_or_p.argmax(dim=-1) == gold).float()


def topk_accuracy(logp_or_p: torch.Tensor, gold: torch.Tensor, k: int = 5) -> torch.Tensor:
    """1 where gold is among the top-k -> (...,) float."""
    topk = logp_or_p.topk(k, dim=-1).indices            # (..., k)
    return (topk == gold.unsqueeze(-1)).any(dim=-1).float()


# --------------------------------------------------------------------------- #
# Confidence / margin (optional difficulty C)
# --------------------------------------------------------------------------- #
def top1_prob_and_margin(p: torch.Tensor):
    """Return (top1_prob, top1-top2 margin), each (...,)."""
    top2 = p.topk(2, dim=-1).values
    return top2[..., 0], top2[..., 0] - top2[..., 1]


# --------------------------------------------------------------------------- #
# Difficulty grouping & rescue (used in the analysis stage)
# --------------------------------------------------------------------------- #
def quantile_groups(d: torch.Tensor, n_groups: int = 4) -> torch.Tensor:
    """Bucket a 1-D difficulty score into ``n_groups`` equal-frequency groups.

    Returns an int tensor in ``[0, n_groups)`` (0 = easiest, n_groups-1 = hardest).
    """
    d = d.flatten()
    qs = torch.linspace(0, 1, n_groups + 1)[1:-1].to(d)
    edges = torch.quantile(d, qs)
    return torch.bucketize(d, edges)


def rescue_scores(h_early: torch.Tensor, h_late: torch.Tensor,
                  nll_early: torch.Tensor = None, nll_late: torch.Tensor = None):
    """Both rescue notions (exp 4).

    * entropy rescue  : H(early) - H(late)        (uncertainty dropped)
    * gold-NLL rescue : NLL(early) - NLL(late)     (confidence in the *correct* token grew)
    Returns a dict of tensors (each shaped like the inputs).
    """
    out = {"rescue_H": h_early - h_late}
    if nll_early is not None and nll_late is not None:
        out["rescue_NLL"] = nll_early - nll_late
    return out


def false_rescue_flag(h_late: torch.Tensor, nll_late: torch.Tensor,
                      h_cut: float = 0.672, nll_cut: float = None) -> torch.Tensor:
    """False rescue: entropy collapsed (confident) but the model is confident in the
    *wrong* token (gold-NLL still high).

    * ``h_cut``   : entropy threshold below which we call the model "confident"
                    (default 0.672 nats, the 2506.01939 forking cutoff).
    * ``nll_cut`` : NLL threshold above which the correct token is "still unlikely".
                    If None, caller should pass e.g. an upper percentile of NLL.
    Returns a float mask.
    """
    confident = (h_late < h_cut).float()
    if nll_cut is None:
        nll_cut = torch.quantile(nll_late.flatten().float(), 0.75).item()
    wrong = (nll_late > nll_cut).float()
    return confident * wrong


# --------------------------------------------------------------------------- #
# Reveal-set builders (which positions are "clean"/visible for a probe forward).
# Each returns a boolean mask of shape (B, L); True = revealed.
# --------------------------------------------------------------------------- #
def random_reveal(B: int, L: int, ratio: float, device="cpu",
                  exclude: torch.Tensor = None, generator=None) -> torch.Tensor:
    """Reveal a random ``ratio`` fraction of positions per row.

    ``exclude`` (B,L bool) positions are forced masked (e.g. the target token).
    """
    scores = torch.rand(B, L, device=device, generator=generator)
    if exclude is not None:
        scores = scores.masked_fill(exclude.bool(), 2.0)   # push excluded to the back
    k = int(round(ratio * L))
    if k <= 0:
        return torch.zeros(B, L, dtype=torch.bool, device=device)
    thresh = scores.kthvalue(k, dim=1, keepdim=True).values
    return scores <= thresh


def _windowed_reveal(target: int, count: int, L: int, mode: str) -> torch.Tensor:
    """Build a 1-D (L,) bool reveal mask of ``count`` positions relative to ``target``.

    mode in {"near", "far", "left", "right", "balanced"}.
    """
    mask = torch.zeros(L, dtype=torch.bool)
    others = [j for j in range(L) if j != target]
    if mode == "near":
        order = sorted(others, key=lambda j: abs(j - target))
        chosen = order[:count]
    elif mode == "far":
        order = sorted(others, key=lambda j: -abs(j - target))
        chosen = order[:count]
    elif mode == "left":
        left = [j for j in others if j < target]
        chosen = sorted(left, key=lambda j: target - j)[:count]   # nearest-left first
    elif mode == "right":
        right = [j for j in others if j > target]
        chosen = sorted(right, key=lambda j: j - target)[:count]
    elif mode == "balanced":
        order = sorted(others, key=lambda j: abs(j - target))
        chosen = order[:count]                                     # symmetric by construction
    else:
        raise ValueError(mode)
    mask[chosen] = True
    return mask


def structured_reveal(targets: torch.Tensor, count: int, L: int, mode: str,
                      device="cpu") -> torch.Tensor:
    """Per-row structured reveal. ``targets`` is (B,) target positions -> (B,L) bool."""
    rows = [_windowed_reveal(int(t), count, L, mode) for t in targets]
    return torch.stack(rows, dim=0).to(device)
