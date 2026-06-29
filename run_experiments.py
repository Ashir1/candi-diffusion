#!/usr/bin/env python
"""Collect data for the denoise-vs-difficulty experiments on the unmodified CANDI model.

Each experiment saves raw per-token tensors under ``--out``; figures are made by ``analyze.py``.
See README_experiments.md for the experiment <-> code <-> figure map.

    python run_experiments.py --exp all --out experiments/out/run1 \
        --n-seqs 64 --steps 256 --batch-size 4 --gold-hf wikitext:wikitext-2-raw-v1:validation

Run inside the candi env (torch + flash-attn + omegaconf) on a GPU.
"""
import argparse
import json
import os

import torch

from denoise_diff import metrics as D, harness as H
from denoise_diff.model import load_model, load_gold, gen_ppl, free


# --- exp 1 & 2: shared sampler trajectory ---------------------------------- #
def run_trace(model, n_seqs, steps, batch_size, seed, out):
    print(f"[exp1/2] tracing {n_seqs} sequences x {steps} steps ...")
    acc = {"final_tokens": [], "reveal_step": [], "committed": [], "H0": [], "H_traj": []}
    done, bi = 0, 0
    while done < n_seqs:
        b = min(batch_size, n_seqs - done)
        tr = H.trace_sample(model, b, num_steps=steps, store_traj=True, seed=seed + bi)
        for k in ("final_tokens", "reveal_step", "committed", "H0"):
            acc[k].append(tr[k])
        acc["H_traj"].append(tr["H_traj"])
        done += b; bi += 1
        print(f"  traced {done}/{n_seqs}")
    data = {k: torch.cat(v, dim=0) for k, v in acc.items() if k != "H_traj"}
    data["H_traj"] = torch.cat(acc["H_traj"], dim=1)          # (S, n_seqs, L)
    data["num_steps"] = steps
    torch.save(data, os.path.join(out, "exp1_2_trace.pt"))
    print(f"  saved {out}/exp1_2_trace.pt  H_traj={tuple(data['H_traj'].shape)}")


# --- exp 3: structured-context probe (reference-free) ---------------------- #
def _probe_targets(model, x_rows, reveal_rows, ratio, targets, batch_size, gold_rows=None):
    """Chunk rows through denoiser_probe; return entropy (and gold metrics) at each row's target."""
    ents, nlls, corr = [], [], []
    for s in range(0, x_rows.size(0), batch_size):
        out = H.denoiser_probe(model, x_rows[s:s + batch_size], reveal_rows[s:s + batch_size], ratio,
                               gold=None if gold_rows is None else gold_rows[s:s + batch_size])
        ts = targets[s:s + batch_size]; idx = torch.arange(ts.size(0))
        ents.append(out["entropy"][idx, ts])
        if gold_rows is not None:
            nlls.append(out["gold_nll"][idx, ts]); corr.append(out["correct"][idx, ts])
    res = {"entropy": torch.cat(ents)}
    if gold_rows is not None:
        res["gold_nll"] = torch.cat(nlls); res["correct"] = torch.cat(corr)
    return res


def run_exp3(model, ref_tokens, ratios, n_targets, batch_size, seed, out, diff_ratios=None):
    print(f"[exp3] near/far/left/right probes on {ref_tokens.size(0)} ref seqs ...")
    torch.manual_seed(seed)
    Bn, L = ref_tokens.shape
    modes = ["near", "far", "left", "right", "balanced"]
    rows_x, rows_t = [], []
    for b in range(Bn):                                       # pick targets per seq (avoid pos 0 = BOS)
        for t in (torch.randperm(L - 1)[:n_targets] + 1):
            rows_x.append(ref_tokens[b]); rows_t.append(int(t))
    x_rows = torch.stack(rows_x, 0); targets = torch.tensor(rows_t)
    res = {"targets": targets, "ratios": ratios, "modes": modes}
    for r in ratios:
        count = int(round(r * L))
        for mode in modes:
            reveal = (torch.zeros_like(x_rows, dtype=torch.bool) if count == 0
                      else D.structured_reveal(targets, count, L, mode))
            res[f"ent_r{int(r*100)}_{mode}"] = _probe_targets(model, x_rows, reveal, r, targets, batch_size)["entropy"]
        print(f"  ratio {r:.2f} done")
    if diff_ratios:                                           # D_i(rho): target entropy under rho% RANDOM context
        res["diff_ratios"] = diff_ratios
        excl = torch.zeros(x_rows.size(0), L, dtype=torch.bool)
        excl[torch.arange(x_rows.size(0)), targets] = True
        for rho in diff_ratios:
            reveal = D.random_reveal(x_rows.size(0), L, rho, exclude=excl)
            res[f"diff_r{int(rho*100)}"] = _probe_targets(model, x_rows, reveal, rho, targets, batch_size)["entropy"]
            print(f"  diff-ref {rho:.2f} done")
    torch.save(res, os.path.join(out, "exp3_structured.pt"))
    print(f"  saved {out}/exp3_structured.pt")


# --- exp 4: rescue / accuracy on gold data --------------------------------- #
def run_exp4(model, gold_tokens, ratios, target_frac, batch_size, seed, out):
    print(f"[exp4] gold-context rescue probes on {gold_tokens.size(0)} seqs ...")
    g = torch.Generator(device=model.device).manual_seed(seed)
    Bn, L = gold_tokens.shape
    target_mask = (torch.rand(Bn, L, device=model.device, generator=g) < target_frac)
    target_mask[:, 0] = False                                 # never target BOS
    rows = {"target_mask": target_mask.cpu(), "ratios": ratios,
            "entropy": [], "gold_nll": [], "correct": [], "top3": []}
    for r in ratios:
        reveal = D.random_reveal(Bn, L, r, device=model.device, exclude=target_mask, generator=g)
        ent, nll, cor, t3 = [], [], [], []
        for s in range(0, Bn, batch_size):
            o = H.denoiser_probe(model, gold_tokens[s:s + batch_size], reveal[s:s + batch_size], r,
                                 gold=gold_tokens[s:s + batch_size])
            ent.append(o["entropy"]); nll.append(o["gold_nll"]); cor.append(o["correct"]); t3.append(o["top3"])
        rows["entropy"].append(torch.cat(ent)); rows["gold_nll"].append(torch.cat(nll))
        rows["correct"].append(torch.cat(cor)); rows["top3"].append(torch.cat(t3))
        print(f"  ratio {r:.2f} done")
    for k in ("entropy", "gold_nll", "correct", "top3"):
        rows[k] = torch.stack(rows[k], 0)                    # (n_ratios, B, L)
    torch.save(rows, os.path.join(out, "exp4_rescue.pt"))
    print(f"  saved {out}/exp4_rescue.pt")


# --- exp 6: rescue along the stochastic reveal schedule (gold) ------------- #
def run_exp6(model, gold_tokens, steps, batch_size, seed, out):
    print(f"[exp6] gold reconstruction on {gold_tokens.size(0)} seqs ...")
    acc = {"reveal_step": [], "nll0": [], "nll_reveal": [], "top3_reveal": [], "top5_reveal": [], "H_traj": []}
    done, bi = 0, 0
    while done < gold_tokens.size(0):
        b = min(batch_size, gold_tokens.size(0) - done)
        o = H.gold_reconstruct_trace(model, gold_tokens[done:done + b], num_steps=steps, seed=seed + bi)
        for k in ("reveal_step", "nll0", "nll_reveal", "top3_reveal", "top5_reveal"):
            acc[k].append(o[k])
        acc["H_traj"].append(o["H_traj"]); done += b; bi += 1
        print(f"  reconstructed {done}/{gold_tokens.size(0)}")
    data = {k: torch.cat(v, 0) for k, v in acc.items() if k != "H_traj"}
    data["H_traj"] = torch.cat(acc["H_traj"], dim=1)          # (S, N, L)
    data["num_steps"] = steps
    torch.save(data, os.path.join(out, "exp6_recon.pt"))
    print(f"  saved {out}/exp6_recon.pt  H_traj={tuple(data['H_traj'].shape)}")


# --- exp 7: gold reconstruction + re-noise intervention (trajectories) ----- #
def run_exp7(model, tok, gold_tokens, renoise_ratios, n_seqs, steps, batch_size, frac, seed, do_ppl, out):
    print("[exp7] gold re-noise & resample with trajectories ...")
    L = model.num_tokens
    conditions = [None, "hard", "random", "easy"]
    traj_keys = ["denoise_ratio", "ent_all", "ent_renoise", "ent_non",
                 "acc3_all", "acc3_renoise", "acc3_non", "acc5_all", "acc5_renoise", "acc5_non"]
    results = {"perturb_ratios": renoise_ratios, "frac": frac, "L": L, "runs": []}
    out_path = os.path.join(out, "exp7_renoise.pt")
    done_keys = set()
    if os.path.exists(out_path):                              # resume
        prev = torch.load(out_path, map_location="cpu", weights_only=False)
        results["runs"] = prev["runs"]
        done_keys = {(e["ratio"], e["selection"]) for e in prev["runs"]}
        print(f"  resuming: {len(done_keys)} (ratio,sel) already on disk")
    for r in renoise_ratios:
        for sel in conditions:
            if (r, str(sel)) in done_keys:
                continue
            trajs, finals, done, bi = [], [], 0, 0
            while done < n_seqs:
                b = min(batch_size, n_seqs - done)
                o = H.gold_renoise_trace(model, gold_tokens[done:done + b], intervention_ratio=r,
                                         selection=sel, frac=frac, num_steps=steps, seed=seed + bi)
                trajs.append(o); finals.append(o["final_tokens"]); done += b; bi += 1
            entry = {"ratio": r, "selection": str(sel), "final_tokens": torch.cat(finals, 0)}
            for k in traj_keys:                               # nanmean trajectories across batches
                entry[k] = torch.nanmean(torch.stack([t[k] for t in trajs], 0), dim=0)
            if do_ppl:
                entry["gen_ppl"] = gen_ppl(model, tok, entry["final_tokens"], L)
            results["runs"].append(entry)
            torch.save(results, out_path)                     # incremental save
            print(f"  ratio {r:.2f} sel={sel} ppl={entry.get('gen_ppl')}")
            free()
    print(f"  saved {out_path}")


def _generate_ref(model, n, steps, batch_size, seed):
    """Self-generate ``n`` reference sequences (for exp3 when run standalone), chunked to avoid OOM."""
    outs, done, bi = [], 0, 0
    while done < n:
        b = min(batch_size, n - done)
        outs.append(H.trace_sample(model, b, num_steps=steps, store_traj=False, seed=seed + bi)["final_tokens"])
        done += b; bi += 1
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="comma list of {1,2,3,4,6,7} or 'all' (1&2 share a trace)")
    ap.add_argument("--ckpt", default="/data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt")
    ap.add_argument("--out", default="experiments/out/run1")
    ap.add_argument("--scratch-dir", default=os.environ.get("SCRATCH_DIR", "/home/patrick/.cache/discrete_diffusion"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--ratios", default="0,20,40,60,80")
    ap.add_argument("--diff-ratios", default="", help="exp3 difficulty-reference reveal %% (e.g. 10,20,30,40)")
    ap.add_argument("--renoise-ratios", default="10,20,30,40", help="exp7 denoise %% to intervene at")
    ap.add_argument("--n-targets", type=int, default=8, help="exp3 target positions per ref sequence")
    ap.add_argument("--target-frac", type=float, default=0.2, help="exp4 fraction of positions kept masked")
    ap.add_argument("--frac", type=float, default=0.2, help="exp7 fraction of committed tokens re-noised")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-ppl", action="store_true", help="exp7: compute GPT-2-large gen-PPL (downloads ~3GB)")
    ap.add_argument("--gold-file", default=None, help="exp4/6/7: gold from a UTF-8 text file")
    ap.add_argument("--gold-hf", default=None, help="exp4/6/7: gold from HF dataset NAME[:CONFIG[:SPLIT]]")
    args = ap.parse_args()

    ratios = [int(x) / 100.0 for x in args.ratios.split(",")]
    renoise_ratios = [int(x) / 100.0 for x in args.renoise_ratios.split(",")]
    diff_ratios = [int(x) / 100.0 for x in args.diff_ratios.split(",")] if args.diff_ratios else None
    exps = ["1", "2", "3", "4", "6", "7"] if args.exp == "all" else args.exp.split(",")
    os.makedirs(args.out, exist_ok=True)

    model, tok, cfg = load_model(args.ckpt, args.device, args.scratch_dir, length=1024)
    json.dump({"args": vars(args), "ratios": ratios}, open(os.path.join(args.out, "meta.json"), "w"),
              indent=2, default=str)

    def _gold():
        return load_gold(model, cfg, tok, args.n_seqs, args.gold_file, args.gold_hf)

    ref_tokens = None
    if "1" in exps or "2" in exps:
        run_trace(model, args.n_seqs, args.steps, args.batch_size, args.seed, args.out)
        ref_tokens = torch.load(os.path.join(args.out, "exp1_2_trace.pt"))["final_tokens"]
        free()

    if "3" in exps:
        n_ref = min(args.n_seqs, 8)
        if ref_tokens is None:
            ref_tokens = _generate_ref(model, n_ref, args.steps, args.batch_size, args.seed)
        run_exp3(model, ref_tokens[:n_ref], ratios, args.n_targets, args.batch_size, args.seed,
                 args.out, diff_ratios=diff_ratios)
        free()

    if "4" in exps:
        try:
            run_exp4(model, _gold(), ratios, args.target_frac, args.batch_size, args.seed, args.out)
        except Exception as e:
            print(f"[exp4] skipped: {e}\n       tip: --gold-hf wikitext:wikitext-2-raw-v1:validation")
        free()

    if "6" in exps:
        try:
            run_exp6(model, _gold(), args.steps, args.batch_size, args.seed, args.out)
        except Exception as e:
            print(f"[exp6] skipped: {e}\n       tip: --gold-hf wikitext:wikitext-2-raw-v1:validation")
        free()

    if "7" in exps:
        try:
            run_exp7(model, tok, _gold(), renoise_ratios, args.n_seqs, args.steps, args.batch_size,
                     args.frac, args.seed, args.gen_ppl, args.out)
        except Exception as e:
            print(f"[exp7] skipped: {e}\n       tip: --gold-hf wikitext:wikitext-2-raw-v1:validation")
        free()

    print("done.")


if __name__ == "__main__":
    main()
