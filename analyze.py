#!/usr/bin/env python
"""Make figures for the denoise-vs-difficulty experiments from the tensors in ``--in``.

Views (``--views``, default all):
  standard  -- exp 1/2 order+collapse, exp 3 context, exp 4 rescue+taxonomy (overlaid)
  per_ratio -- exp 1/2/3/4 as grouped bars at each denoise %
  difftime  -- exp 1/2/3/4 re-bucketed by entropy at --diff-ratios (e.g. 10,20,30,40)
  exp6      -- exp 6 rescue scatter/taxonomy + "more time -> rescue", per --diff-ratios
  exp7      -- exp 7 per bucketing %: entropy + top-3/top-5 over time + gen-PPL

  python analyze.py --in experiments/out/run_big --figdir gen_imgs/run_big --diff-ratios 10,20,30,40
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from denoise_diff import metrics as M, plotting as P

RATIOS_DEFAULT = [0.0, 0.2, 0.4, 0.6, 0.8]


# =========================================================================== #
# standard view (overlaid figures)
# =========================================================================== #
def standard(indir, figdir):
    d = P.load(indir, "exp1_2_trace.pt")
    if d is not None:
        H, rs, H0 = d["H_traj"].float(), d["reveal_step"].long(), d["H0"].float()
        S, B, L = H.shape
        bb = torch.arange(B)[:, None].expand(B, L); ll = torch.arange(L)[None, :].expand(B, L)
        H_at = H[rs.clamp(0, S - 1), bb, ll]; reveal_frac = rs.float() / max(S - 1, 1)
        steps, diff = [], []
        for i in range(S):
            c, m = (rs == i), (rs >= i)
            if c.sum() > 0 and m.sum() > 0:
                steps.append(i / max(S - 1, 1)); diff.append((H[i][c].mean() - H[i][m].mean()).item())
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        sp, pr = P.corr(reveal_frac.flatten(), H_at.flatten())
        ax[0].hexbin(reveal_frac.flatten(), H_at.flatten(), gridsize=40, cmap="viridis", mincnt=1)
        ax[0].set(title=f"reveal frac vs entropy-at-reveal\nSpearman={sp:.3f} Pearson={pr:.3f}",
                  xlabel="reveal fraction (tau/S)", ylabel="entropy at reveal (nats)")
        sp0, _ = P.corr(reveal_frac.flatten(), H0.flatten())
        ax[1].hexbin(reveal_frac.flatten(), H0.flatten(), gridsize=40, cmap="magma", mincnt=1)
        ax[1].set(title=f"NULL CONTROL: reveal frac vs H0\nSpearman={sp0:.3f} (expect ~0)",
                  xlabel="reveal fraction", ylabel="H0 (nats)")
        ax[2].axhline(0, color="k", lw=0.8); ax[2].plot(steps, diff, ".", ms=3, alpha=0.6)
        ax[2].set(title="committed - masked-cohort entropy per step\n(expect ~0 => order indep. of difficulty)",
                  xlabel="reveal fraction", ylabel="mean entropy diff (nats)")
        P.save_fig(fig, figdir, "exp1_order_vs_difficulty.png")

        groups = M.quantile_groups(H0, 4).reshape(B, L)
        rps = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
        fig, ax = plt.subplots(figsize=(7, 5))
        for g in range(4):
            ys = [H[i][(rs > i) & (groups == g)].mean().item() if ((rs > i) & (groups == g)).sum() else np.nan
                  for i in range(S)]
            ax.plot(rps.numpy(), ys, label=f"D_i q{g} ({'easy' if g==0 else 'hard' if g==3 else ''})")
        ax.set(title="Exp2: entropy collapse vs reveal ratio", xlabel="reveal ratio", ylabel="entropy (nats)")
        ax.legend(); P.save_fig(fig, figdir, "exp2_entropy_collapse.png")

    d = P.load(indir, "exp3_structured.pt")
    if d is not None:
        ratios, modes = d["ratios"], d["modes"]
        groups = M.quantile_groups(d[f"ent_r{int(ratios[0]*100)}_{modes[0]}"].float(), 4)
        fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), sharey=True)
        for g in range(4):
            for mode in modes:
                axes[g].plot(ratios, [d[f"ent_r{int(r*100)}_{mode}"].float()[groups == g].mean().item()
                                      for r in ratios], marker="o", label=mode)
            axes[g].set(title=f"D_i quartile {g}", xlabel="reveal ratio")
            axes[g].legend(fontsize=8)
        axes[0].set_ylabel("entropy at target (nats)")
        fig.suptitle("Exp3: which context collapses entropy (near/far/left/right), by difficulty")
        P.save_fig(fig, figdir, "exp3_structured_context.png")

    d = P.load(indir, "exp4_rescue.pt")
    if d is not None:
        ratios, tm = d["ratios"], d["target_mask"].bool()
        R = d["entropy"].shape[0]; sel = tm.unsqueeze(0).expand(R, -1, -1)
        H = d["entropy"].float()[sel].reshape(R, -1); N = d["gold_nll"].float()[sel].reshape(R, -1)
        C = d["correct"].float()[sel].reshape(R, -1)
        hard = M.quantile_groups(H[0], 2) == 1
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for lbl, msk in [("easy", ~hard), ("hard", hard)]:
            ax[0].plot(ratios, [H[i][msk].mean().item() for i in range(R)], marker="o", color=P.COLORS[lbl], label=lbl)
            ax[1].plot(ratios, [N[i][msk].mean().item() for i in range(R)], marker="o", color=P.COLORS[lbl], label=lbl)
        ax[0].set(title="Exp4: entropy collapse", xlabel="reveal ratio", ylabel="entropy (nats)"); ax[0].legend()
        ax[1].set(title="Exp4: gold-NLL collapse", xlabel="reveal ratio", ylabel="gold NLL (nats)"); ax[1].legend()
        P.save_fig(fig, figdir, "exp4_rescue_curves.png")

        rescue_H, rescue_N = H[0] - H[-1], N[0] - N[-1]
        nll_cut = torch.quantile(N[-1], 0.75).item()
        false_rescue = (H[-1] < P.FORK_CUT) & (N[-1] > nll_cut)
        correct, rescued = C[-1] > 0.5, rescue_N > 0
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
        ax[0].scatter(rescue_H[~false_rescue], rescue_N[~false_rescue], s=6, alpha=0.4, label="normal")
        ax[0].scatter(rescue_H[false_rescue], rescue_N[false_rescue], s=10, color="red", alpha=0.7, label="false rescue")
        ax[0].set(title="Exp4: entropy-rescue vs gold-NLL-rescue\n(false rescue = confident but wrong)",
                  xlabel="entropy rescue H0-Hlate", ylabel="gold-NLL rescue NLL0-NLLlate"); ax[0].legend()
        cells, masses = [], []
        for hl, hm in [("hard", hard), ("easy", ~hard)]:
            for rl, rm in [("rescued", rescued), ("not", ~rescued)]:
                for cl, cm in [("correct", correct), ("wrong", ~correct)]:
                    cells.append(f"{hl}\n{rl}\n{cl}"); masses.append(float((hm & rm & cm).float().mean()))
        ax[1].bar(range(8), masses); ax[1].set_xticks(range(8)); ax[1].set_xticklabels(cells, fontsize=6)
        ax[1].set(title="Exp4: hard/easy x rescued/not x correct/wrong", ylabel="fraction")
        P.save_fig(fig, figdir, "exp4_rescue_taxonomy.png")
        print(f"  false-rescue fraction: {false_rescue.float().mean().item():.3f}")


# =========================================================================== #
# per_ratio view (grouped bars at each denoise %)
# =========================================================================== #
def per_ratio(indir, figdir, ratios=RATIOS_DEFAULT):
    pcts = [f"{int(r*100)}%" for r in ratios]
    d = P.load(indir, "exp1_2_trace.pt")
    if d is not None:
        H, rs, H0 = d["H_traj"].float(), d["reveal_step"].long(), d["H0"].float()
        S, B, L = H.shape
        rps = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
        star = [int(torch.argmin((rps - r).abs())) for r in ratios]
        groups = M.quantile_groups(H0, 4).reshape(B, L)
        comm = [H[i][rs == i].mean().item() if (rs == i).sum() else np.nan for i in star]
        coh = [H[i][rs > i].mean().item() if (rs > i).sum() else np.nan for i in star]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        P.grouped_bars(ax, pcts, {"just-committed": comm, "still-masked cohort": coh}, "entropy (nats)",
                       "Exp1 per ratio: committed vs masked-cohort entropy\n(equal => order indep. of difficulty)",
                       annotate=True)
        P.save_fig(fig, figdir, "exp1_per_ratio.png")
        series = {}
        for g in range(4):
            series[f"D_i q{g}" + ("(easy)" if g == 0 else "(hard)" if g == 3 else "")] = [
                H[i][(rs > i) & (groups == g)].mean().item() if ((rs > i) & (groups == g)).sum() else np.nan
                for i in star]
        fig, ax = plt.subplots(figsize=(9, 4.5))
        P.grouped_bars(ax, pcts, series, "entropy (nats)",
                       "Exp2 per ratio: still-masked entropy by initial difficulty")
        P.save_fig(fig, figdir, "exp2_per_ratio.png")

    d = P.load(indir, "exp3_structured.pt")
    if d is not None:
        rs_, modes = d["ratios"], d["modes"]
        hard = M.quantile_groups(d[f"ent_r{int(rs_[0]*100)}_{modes[0]}"].float(), 2) == 1
        fig, axes = plt.subplots(1, len(rs_), figsize=(4.2 * len(rs_), 4.5), sharey=True)
        for ax, r in zip(np.atleast_1d(axes), rs_):
            series = {lbl: [d[f"ent_r{int(r*100)}_{m}"].float()[msk].mean().item() for m in modes]
                      for lbl, msk in [("easy", ~hard), ("hard", hard)]}
            P.grouped_bars(ax, modes, series, "entropy at target (nats)", f"{int(r*100)}% revealed")
        fig.suptitle("Exp3 per ratio: context type (near/far/left/right) by easy/hard")
        P.save_fig(fig, figdir, "exp3_per_ratio.png")

    d = P.load(indir, "exp4_rescue.pt")
    if d is not None:
        rs_, tm = d["ratios"], d["target_mask"].bool(); R = d["entropy"].shape[0]
        sel = tm.unsqueeze(0).expand(R, -1, -1)
        Ht = d["entropy"].float()[sel].reshape(R, -1); Nt = d["gold_nll"].float()[sel].reshape(R, -1)
        At = d["correct"].float()[sel].reshape(R, -1)
        hard = M.quantile_groups(Ht[0], 2) == 1
        panels = [(Ht, "entropy (nats)", "entropy"), (Nt, "gold NLL (nats)", "gold-NLL"),
                  (At, "accuracy", "top-1 accuracy")]
        if "top3" in d:
            panels.append((d["top3"].float()[sel].reshape(R, -1), "accuracy", "top-3 accuracy"))
        fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.5))
        for ax, (Mt, ylab, title) in zip(np.atleast_1d(axes), panels):
            series = {"easy": [Mt[i][~hard].mean().item() for i in range(R)],
                      "hard": [Mt[i][hard].mean().item() for i in range(R)]}
            P.grouped_bars(ax, [f"{int(r*100)}%" for r in rs_], series, ylab, f"Exp4 per ratio: {title}")
        P.save_fig(fig, figdir, "exp4_per_ratio.png")


# =========================================================================== #
# difftime view (re-bucket by entropy at rho)
# =========================================================================== #
def difftime(indir, figdir, rho):
    d = P.load(indir, "exp1_2_trace.pt")
    if d is not None:
        H, rs = d["H_traj"].float(), d["reveal_step"].long(); S, B, L = H.shape
        rps = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
        s0 = int(torch.argmin((rps - rho).abs())); masked = rs > s0
        if masked.sum() >= 20:
            Dv = H[s0][masked]; groups = torch.full((B, L), -1, dtype=torch.long); groups[masked] = M.quantile_groups(Dv, 4)
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
            xs = rps[s0:].numpy()
            for g in range(4):
                ys = [H[s][(rs > s) & (groups == g)].mean().item() if ((rs > s) & (groups == g)).sum() else np.nan
                      for s in range(s0, S)]
                ax[0].plot(xs, ys, label=f"q{g}" + ("(easy)" if g == 0 else "(hard)" if g == 3 else ""))
            ax[0].set(title=f"Exp2 | difficulty @ {int(rho*100)}%: entropy collapse of tokens masked at {int(rho*100)}%",
                      xlabel="reveal ratio", ylabel="entropy (nats)"); ax[0].legend(fontsize=8)
            later = rs[masked].float() / max(S - 1, 1)
            sp, _ = P.corr(Dv.numpy(), later.numpy())
            ax[1].hexbin(later.numpy(), Dv.numpy(), gridsize=35, cmap="magma", mincnt=1)
            ax[1].set(title=f"Exp1 null @ {int(rho*100)}%: reveal order vs D_i(rho)\nSpearman={sp:.3f} (expect ~0)",
                      xlabel="reveal fraction (of tokens masked at rho)", ylabel=f"entropy at {int(rho*100)}% (nats)")
            P.save_fig(fig, figdir, f"exp1_2_difftime_diff{int(rho*100)}.png")

    d = P.load(indir, "exp3_structured.pt")
    if d is not None and f"diff_r{int(rho*100)}" in d:
        ratios, modes = d["ratios"], d["modes"]
        hard = M.quantile_groups(d[f"diff_r{int(rho*100)}"].float(), 2) == 1
        fig, axes = plt.subplots(1, len(ratios), figsize=(4.2 * len(ratios), 4.5), sharey=True)
        for ax, r in zip(np.atleast_1d(axes), ratios):
            series = {lbl: [d[f"ent_r{int(r*100)}_{m}"].float()[msk].mean().item() for m in modes]
                      for lbl, msk in [("easy", ~hard), ("hard", hard)]}
            P.grouped_bars(ax, modes, series, "entropy at target (nats)", f"{int(r*100)}% revealed")
        fig.suptitle(f"Exp3 | difficulty @ {int(rho*100)}% denoised: context type by easy/hard")
        P.save_fig(fig, figdir, f"exp3_difftime_diff{int(rho*100)}.png")

    d = P.load(indir, "exp4_rescue.pt")
    if d is not None:
        ratios, tm = d["ratios"], d["target_mask"].bool(); R = d["entropy"].shape[0]
        iro = int(np.argmin([abs(r - rho) for r in ratios])); sel = tm.unsqueeze(0).expand(R, -1, -1)
        Ht = d["entropy"].float()[sel].reshape(R, -1); Nt = d["gold_nll"].float()[sel].reshape(R, -1)
        At = d["correct"].float()[sel].reshape(R, -1)
        hard = M.quantile_groups(Ht[iro], 2) == 1
        panels = [(Ht, "entropy (nats)", "entropy"), (Nt, "gold NLL (nats)", "gold-NLL"),
                  (At, "accuracy", "top-1 accuracy")]
        if "top3" in d:
            panels.append((d["top3"].float()[sel].reshape(R, -1), "accuracy", "top-3 accuracy"))
        pcts = [f"{int(r*100)}%" for r in ratios]
        fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.5))
        for ax, (Mt, ylab, ttl) in zip(np.atleast_1d(axes), panels):
            P.grouped_bars(ax, pcts, {"easy": [Mt[i][~hard].mean().item() for i in range(R)],
                                      "hard": [Mt[i][hard].mean().item() for i in range(R)]},
                           ylab, f"{ttl} (difficulty @ {int(rho*100)}%)")
        fig.suptitle(f"Exp4 | difficulty bucketed by entropy at {int(rho*100)}% gold context")
        P.save_fig(fig, figdir, f"exp4_difftime_diff{int(rho*100)}.png")

        fwd = [i for i, r in enumerate(ratios) if r >= rho - 1e-6]; xs = [ratios[i] for i in fwd]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.3))
        for ax, (Mt, ylab, ttl) in zip(np.atleast_1d(axes), panels):
            ax.plot(xs, [Mt[i][~hard].mean().item() for i in fwd], marker="o", color=P.COLORS["easy"], label="easy")
            ax.plot(xs, [Mt[i][hard].mean().item() for i in fwd], marker="o", color=P.COLORS["hard"], label="hard")
            ax.axvline(rho, color="k", ls=":", lw=0.8); ax.set(xlabel="reveal ratio", ylabel=ylab, title=ttl); ax.legend(fontsize=8)
        fig.suptitle(f"Exp4 forward from {int(rho*100)}%: easy vs hard")
        P.save_fig(fig, figdir, f"exp4_difftime_fwd_diff{int(rho*100)}.png")


# =========================================================================== #
# exp6 view (gold reconstruction rescue + "more time -> rescue")
# =========================================================================== #
def exp6(indir, figdir, rho):
    d = P.load(indir, "exp6_recon.pt")
    if d is None:
        return
    H, rs = d["H_traj"].float(), d["reveal_step"].long()
    nll0, nllr = d["nll0"].float(), d["nll_reveal"].float()
    t3, t5 = d["top3_reveal"].float(), d["top5_reveal"].float()
    S, N, L = H.shape
    rps = torch.tensor([(rs < i).float().mean().item() for i in range(S)])
    s0 = int(torch.argmin((rps - rho).abs())); masked = rs > s0
    if masked.sum() < 30:
        return
    rsc = rs.clamp(0, S - 1); nn = torch.arange(N)[:, None].expand(N, L); ll = torch.arange(L)[None, :].expand(N, L)
    H_rev = H[rsc, nn, ll]; m = masked
    Dv = H[s0][m]; hard = Dv > Dv.median()
    rescue_H, rescue_N = (H[0] - H_rev)[m], (nll0 - nllr)[m]
    t3m, t5m, H_revm = t3[m], t5[m], H_rev[m]
    rescued, correct = rescue_N > 0, t5m > 0.5
    false_rescue = (H_revm < P.FORK_CUT) & (t5m < 0.5)
    ctx = rps[rs[m]]

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].axhline(0, color="k", lw=0.6); ax[0].axvline(0, color="k", lw=0.6)
    ax[0].scatter(rescue_H[~false_rescue], rescue_N[~false_rescue], s=6, alpha=0.35, label="normal")
    ax[0].scatter(rescue_H[false_rescue], rescue_N[false_rescue], s=12, color="red", alpha=0.8, label="false rescue")
    ax[0].set(title=f"Exp6 @ {int(rho*100)}%: entropy-rescue vs gold-NLL-rescue\n(false = confident but not top-5)",
              xlabel="entropy rescue H0-H(reveal)", ylabel="gold-NLL rescue NLL0-NLL(reveal)"); ax[0].legend()
    cells, mass = [], []
    for hl, hm in [("hard", hard), ("easy", ~hard)]:
        for rl, rm in [("rescued", rescued), ("not", ~rescued)]:
            for cl, cm in [("correct", correct), ("wrong", ~correct)]:
                cells.append(f"{hl}\n{rl}\n{cl}"); mass.append(float((hm & rm & cm).float().mean()))
    ax[1].bar(range(8), mass); ax[1].set_xticks(range(8)); ax[1].set_xticklabels(cells, fontsize=6)
    ax[1].set(title=f"Exp6 @ {int(rho*100)}%: hard/easy x rescued/not x correct/wrong (top-5)", ylabel="fraction")
    P.save_fig(fig, figdir, f"exp6_rescue_diff{int(rho*100)}.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    edges = np.linspace(float(ctx.min()), float(ctx.max()), 7); centers = 0.5 * (edges[:-1] + edges[1:])
    for grp, gm, col in [("hard", hard, P.COLORS["hard"]), ("easy", ~hard, P.COLORS["easy"])]:
        c = ctx[gm].numpy()
        for acc, lbl, ls in [(t5m[gm].numpy(), "top-5", "-"), (t3m[gm].numpy(), "top-3", "--")]:
            ys = [acc[(c >= edges[i]) & (c < edges[i + 1] + 1e-6)].mean()
                  if ((c >= edges[i]) & (c < edges[i + 1] + 1e-6)).sum() else np.nan for i in range(len(edges) - 1)]
            ax.plot(centers, ys, ls, color=col, marker="o", label=f"{grp} {lbl}")
    ax.set(title=f"Exp6 @ {int(rho*100)}%: more reveal-time -> rescue (tokens masked at {int(rho*100)}%)",
           xlabel="context revealed when token committed", ylabel="fraction correct vs gold"); ax.legend(fontsize=8)
    P.save_fig(fig, figdir, f"exp6_time_vs_rescue_diff{int(rho*100)}.png")


# =========================================================================== #
# exp7 view (gold re-noise; one figure per bucketing %)
# =========================================================================== #
def exp7(indir, figdir):
    d = P.load(indir, "exp7_renoise.pt")
    if d is None:
        return
    conds = [("hard", P.COLORS["hard"]), ("random", P.COLORS["random"]), ("easy", P.COLORS["easy"])]
    for r in sorted({e["ratio"] for e in d["runs"]}):
        runs = {e["selection"]: e for e in d["runs"] if e["ratio"] == r}
        if "None" not in runs:
            continue
        base = runs["None"]
        fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
        ax[0].plot(P.np_(base["denoise_ratio"]), P.np_(base["ent_all"]), "k:", lw=1.2, label="baseline (all masked)")
        for c, col in conds:
            if c in runs:
                ax[0].plot(P.np_(runs[c]["denoise_ratio"]), P.np_(runs[c]["ent_renoise"]), "-", color=col, label=f"{c} re-noised")
                ax[0].plot(P.np_(runs[c]["denoise_ratio"]), P.np_(runs[c]["ent_non"]), "--", color=col, alpha=0.4)
        ax[0].set(title="entropy of still-masked tokens", xlabel="denoise ratio", ylabel="entropy (nats)"); ax[0].legend(fontsize=7)
        for ax_i, key, lab in [(ax[1], "acc5", "top-5"), (ax[2], "acc3", "top-3")]:
            for c, col in conds:
                if c in runs:
                    ax_i.plot(P.np_(runs[c]["denoise_ratio"]), P.np_(runs[c][f"{key}_renoise"]), "-", color=col, label=c)
            ax_i.plot(P.np_(base["denoise_ratio"]), P.np_(base[f"{key}_all"]), "k:", lw=1.2, label="baseline")
            ax_i.set(title=f"{lab} accuracy of re-noised tokens (vs gold)", xlabel="denoise ratio", ylabel=f"{lab} accuracy")
            ax_i.legend(fontsize=7)
        vals = [runs.get(k, {}).get("gen_ppl") for k in ["None", "hard", "random", "easy"]]
        ax[3].bar(range(4), [v if v is not None else np.nan for v in vals], color=["gray", "C3", "C1", "C0"])
        if runs["None"].get("gen_ppl") is not None:
            ax[3].axhline(runs["None"]["gen_ppl"], color="gray", ls=":", lw=1)
        ax[3].set_xticks(range(4)); ax[3].set_xticklabels(["baseline", "hard", "random", "easy"])
        ax[3].set(title="final gen-PPL (gold + re-sampled subset)", ylabel="gen PPL")
        for a in ax[:3]:
            a.axvline(r, color="gray", ls=":", lw=0.8)
        fig.suptitle(f"Exp7 re-noise @ {int(r*100)}% denoised -- re-noised solid, non-renoised dashed")
        P.save_fig(fig, figdir, f"exp7_renoise_diff{int(r*100)}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="experiments/out/run_big")
    ap.add_argument("--figdir", default="gen_imgs/run_big")
    ap.add_argument("--views", default="all", help="comma list of {standard,per_ratio,difftime,exp6,exp7} or 'all'")
    ap.add_argument("--diff-ratios", default="10,20,30,40")
    a = ap.parse_args()
    views = ["standard", "per_ratio", "difftime", "exp6", "exp7"] if a.views == "all" else a.views.split(",")
    rhos = [int(x) / 100.0 for x in a.diff_ratios.split(",")]
    if "standard" in views:
        print("[standard]"); standard(a.indir, a.figdir)
    if "per_ratio" in views:
        print("[per_ratio]"); per_ratio(a.indir, a.figdir)
    if "difftime" in views:
        for rho in rhos:
            print(f"[difftime {int(rho*100)}%]"); difftime(a.indir, a.figdir, rho)
    if "exp6" in views:
        for rho in rhos:
            print(f"[exp6 {int(rho*100)}%]"); exp6(a.indir, a.figdir, rho)
    if "exp7" in views:
        print("[exp7]"); exp7(a.indir, a.figdir)
    print("done.")


if __name__ == "__main__":
    main()
