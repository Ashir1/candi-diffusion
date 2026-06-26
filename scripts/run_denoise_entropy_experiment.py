#!/usr/bin/env python
"""Driver for the denoise-vs-difficulty experiment suite (see denoise_vs_difficulty_plan.md).

Loads the trained CANDI checkpoint (unmodified model) and runs experiments 1-5,
saving raw per-token tensors under ``--out``. Analysis/plots are produced separately
by ``scripts/analyze_denoise_entropy.py``.

Example
-------
    python scripts/run_denoise_entropy_experiment.py --exp all \
        --ckpt /data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt \
        --out experiments/out/run1 --n-seqs 32 --steps 256 --batch-size 8

Notes
-----
* Run inside the candi env (torch 2.3 + hydra + flash-attn). Needs a GPU.
* Exp 4 needs OpenWebText validation data reachable via ``--scratch-dir`` (dir holding ``owt/``).
"""
import argparse
import gc
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import difficulty as D                       # noqa: E402
import denoise_harness as H                  # noqa: E402


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(ckpt, device, scratch_dir=None, length=None):
    """Load CANDI from the config saved *inside* the checkpoint.

    Needs only ``omegaconf`` (mandatory anyway -- the saved config is an OmegaConf
    pickle), **not** hydra. The trained config already carries the right knobs
    (step_size=0.5, sigma_min/max=0.1/4.0); we only top up any algo keys that the
    current code expects but the (older) trained schema lacks (temp/sampler/
    pure_continuous/is_embed), pulling those from the repo's candi.yaml.
    """
    from omegaconf import OmegaConf, open_dict
    for name, fn in [("cwd", os.getcwd), ("device_count", torch.cuda.device_count),
                     ("eval", eval), ("div_up", lambda x, y: (x + y - 1) // y)]:
        try:
            OmegaConf.register_new_resolver(name, fn)
        except Exception:
            pass  # already registered

    print("[load] reading config from checkpoint (omegaconf only; no hydra) ...")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["hyper_parameters"]["config"]
    repo_algo = OmegaConf.load(os.path.join(REPO, "configs", "algo", "candi.yaml"))
    with open_dict(cfg):
        for k in repo_algo:                       # fill missing keys, keep trained values
            if k not in cfg.algo:
                cfg.algo[k] = repo_algo[k]
        if scratch_dir is not None:
            cfg.scratch_dir = scratch_dir
        if length is not None:
            cfg.model.length = length

    import algo
    import dataloader
    tok = dataloader.get_tokenizer(cfg)
    model = algo.CANDI(cfg, tokenizer=tok)
    sd = {k: v for k, v in ck["state_dict"].items() if "_orig_mod" not in k}   # drop torch.compile keys
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] {len(sd)} tensors | missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    model.ema = None                              # use trained weights directly (see HOWTO for EMA)
    try:
        model.metrics.to(device)
    except Exception:
        pass
    if not hasattr(model, "num_tokens"):
        model.num_tokens = cfg.model.length
    return model, tok, cfg


# --------------------------------------------------------------------------- #
# Experiments 1 & 2 -- shared sampler trajectory (H1)
# --------------------------------------------------------------------------- #
def run_trace(model, n_seqs, steps, batch_size, seed, out):
    print(f"[exp1/2] tracing {n_seqs} sequences x {steps} steps ...")
    acc = {"final_tokens": [], "reveal_step": [], "committed": [], "H0": [], "H_traj": []}
    done = 0
    bi = 0
    while done < n_seqs:
        b = min(batch_size, n_seqs - done)
        tr = H.trace_sample(model, b, num_steps=steps, store_traj=True, seed=seed + bi)
        for k in ("final_tokens", "reveal_step", "committed", "H0"):
            acc[k].append(tr[k])
        acc["H_traj"].append(tr["H_traj"])    # (S,b,L)
        done += b
        bi += 1
        print(f"  traced {done}/{n_seqs}")
    data = {k: torch.cat(v, dim=0) for k, v in acc.items() if k != "H_traj"}
    data["H_traj"] = torch.cat(acc["H_traj"], dim=1)          # (S, n_seqs, L)
    data["num_steps"] = steps
    torch.save(data, os.path.join(out, "exp1_2_trace.pt"))
    print(f"  saved {out}/exp1_2_trace.pt  H_traj={tuple(data['H_traj'].shape)}")


# --------------------------------------------------------------------------- #
# Experiment 3 -- structured-context probe (H2, reference-free)
# --------------------------------------------------------------------------- #
def _probe_targets(model, x_rows, reveal_rows, ratio, targets, batch_size, gold_rows=None):
    """Chunk rows through denoiser_probe; return entropy (and gold metrics) at each row's target."""
    ents, nlls, corr = [], [], []
    for s in range(0, x_rows.size(0), batch_size):
        xs = x_rows[s:s + batch_size]
        rs = reveal_rows[s:s + batch_size]
        ts = targets[s:s + batch_size]
        g = None if gold_rows is None else gold_rows[s:s + batch_size]
        out = H.denoiser_probe(model, xs, rs, ratio, gold=g)
        idx = torch.arange(xs.size(0))
        ents.append(out["entropy"][idx, ts])
        if g is not None:
            nlls.append(out["gold_nll"][idx, ts])
            corr.append(out["correct"][idx, ts])
    res = {"entropy": torch.cat(ents)}
    if gold_rows is not None:
        res["gold_nll"] = torch.cat(nlls)
        res["correct"] = torch.cat(corr)
    return res


def run_exp3(model, ref_tokens, ratios, n_targets, batch_size, seed, out, diff_ratios=None):
    print(f"[exp3] near/far/left/right probes on {ref_tokens.size(0)} ref seqs ...")
    torch.manual_seed(seed)
    Bn, L = ref_tokens.shape
    modes = ["near", "far", "left", "right", "balanced"]
    # pick targets per sequence (avoid pos 0 = BOS)
    rows_x, rows_t = [], []
    for b in range(Bn):
        tgts = torch.randperm(L - 1)[:n_targets] + 1
        for t in tgts:
            rows_x.append(ref_tokens[b])
            rows_t.append(int(t))
    x_rows = torch.stack(rows_x, 0)
    targets = torch.tensor(rows_t)
    res = {"targets": targets, "ratios": ratios, "modes": modes}
    for r in ratios:
        count = int(round(r * L))
        for mode in modes:
            if count == 0:
                reveal = torch.zeros_like(x_rows, dtype=torch.bool)
            else:
                reveal = D.structured_reveal(targets, count, L, mode)
            pr = _probe_targets(model, x_rows, reveal, r, targets, batch_size)
            res[f"ent_r{int(r*100)}_{mode}"] = pr["entropy"]
        print(f"  ratio {r:.2f} done")
    # difficulty score D_i(rho): target entropy under rho% RANDOM context (self-gen regime)
    if diff_ratios:
        res["diff_ratios"] = diff_ratios
        excl = torch.zeros(x_rows.size(0), L, dtype=torch.bool)
        excl[torch.arange(x_rows.size(0)), targets] = True
        for rho in diff_ratios:
            reveal = D.random_reveal(x_rows.size(0), L, rho, exclude=excl)
            pr = _probe_targets(model, x_rows, reveal, rho, targets, batch_size)
            res[f"diff_r{int(rho*100)}"] = pr["entropy"]
            print(f"  diff-ref {rho:.2f} done")
    torch.save(res, os.path.join(out, "exp3_structured.pt"))
    print(f"  saved {out}/exp3_structured.pt")


# --------------------------------------------------------------------------- #
# Experiment 4 -- rescue / false-rescue on gold data (H2, the only GT experiment)
# --------------------------------------------------------------------------- #
def run_exp4(model, gold_tokens, ratios, target_frac, batch_size, seed, out):
    print(f"[exp4] gold-context rescue probes on {gold_tokens.size(0)} validation seqs ...")
    g = torch.Generator(device=model.device).manual_seed(seed)
    Bn, L = gold_tokens.shape
    device = model.device
    # fixed target set kept masked at ALL ratios (so a token is tracked across context budgets)
    target_mask = (torch.rand(Bn, L, device=device, generator=g) < target_frac)
    target_mask[:, 0] = False                                # never target BOS
    rows = {"target_mask": target_mask.cpu(), "ratios": ratios,
            "entropy": [], "gold_nll": [], "correct": [], "top3": []}
    for r in ratios:
        # reveal r-fraction of the *non-target* positions (gold), keep targets masked
        reveal = D.random_reveal(Bn, L, r, device=device, exclude=target_mask, generator=g)
        ent, nll, cor, t3 = [], [], [], []
        for s in range(0, Bn, batch_size):
            out_p = H.denoiser_probe(model, gold_tokens[s:s + batch_size],
                                     reveal[s:s + batch_size], r,
                                     gold=gold_tokens[s:s + batch_size])
            ent.append(out_p["entropy"]); nll.append(out_p["gold_nll"])
            cor.append(out_p["correct"]); t3.append(out_p["top3"])
        rows["entropy"].append(torch.cat(ent))
        rows["gold_nll"].append(torch.cat(nll))
        rows["correct"].append(torch.cat(cor))
        rows["top3"].append(torch.cat(t3))
        print(f"  ratio {r:.2f} done")
    for k in ("entropy", "gold_nll", "correct", "top3"):
        rows[k] = torch.stack(rows[k], 0)                    # (n_ratios, B, L)
    torch.save(rows, os.path.join(out, "exp4_rescue.pt"))
    print(f"  saved {out}/exp4_rescue.pt")


# --------------------------------------------------------------------------- #
# Experiment 5 -- perturb-and-continue (replace with wrong token)
# --------------------------------------------------------------------------- #
def run_exp5(model, tok, perturb_ratios, n_seqs, batch_size, frac, seed, gen_ppl, out):
    print("[exp5] perturb-and-continue (hard/random/easy vs baseline) ...")
    L = model.num_tokens
    conditions = [None, "hard", "random", "easy"]
    results = {"perturb_ratios": perturb_ratios, "frac": frac, "L": L, "runs": []}
    for r in perturb_ratios:
        for sel in conditions:
            finals, pmask = [], []
            done, bi = 0, 0
            while done < n_seqs:
                b = min(batch_size, n_seqs - done)
                o = H.perturb_and_continue(model, b, ratio_r=r, selection=sel,
                                           frac=frac, seed=seed + bi)
                finals.append(o["final_tokens"]); pmask.append(o["perturb_mask"])
                done += b; bi += 1
            finals = torch.cat(finals, 0); pmask = torch.cat(pmask, 0)
            entry = {"ratio": r, "selection": str(sel),
                     "final_tokens": finals, "perturb_mask": pmask}
            if gen_ppl:
                entry["gen_ppl"] = _gen_ppl(model, tok, finals, L)
            results["runs"].append(entry)
            torch.save(results, os.path.join(out, "exp5_perturb.pt"))   # incremental (overnight-safe)
            print(f"  ratio {r:.2f} sel={sel} ppl={entry.get('gen_ppl')}")
            _free()
    print(f"  saved {out}/exp5_perturb.pt")


def _gen_ppl(model, tok, tokens, L):
    try:
        texts = tok.batch_decode(tokens)
        model.metrics.gen_ppl.reset()
        model.metrics.record_generative_perplexity(texts, max_length=L, retokenize=True,
                                                    device=str(model.device))
        return float(model.metrics.gen_ppl.compute().item())
    except Exception as e:                                   # gpt2-large download / OOM etc.
        print(f"  [warn] gen_ppl failed: {e}")
        return None
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
def _load_gold(model, cfg, tok, n_seqs, gold_file=None, gold_hf=None):
    """Gold sequences for exp 4. Three sources (pick whichever you can run):

      --gold-file PATH   any UTF-8 text file (tokenized + chunked to length L)
      --gold-hf NAME[:CONFIG[:SPLIT]]   a HF dataset, e.g. 'wikitext:wikitext-2-raw-v1:validation'
      (default)          OpenWebText via the repo dataloader (needs --scratch-dir/owt cache)
    """
    L = model.num_tokens

    def _chunk(ids):
        ids = torch.tensor(ids, dtype=torch.long)
        n = ids.numel() // L
        if n == 0:
            raise ValueError(f"text too short to fill one sequence of length {L}")
        return ids[: n * L].reshape(n, L)

    if gold_file:
        print(f"[exp4] gold from text file {gold_file}")
        text = open(gold_file, encoding="utf-8", errors="ignore").read()
        seqs = _chunk(tok(text)["input_ids"])
    elif gold_hf:
        from datasets import load_dataset
        parts = (gold_hf.split(":") + [None, None])[:3]
        name, conf, split = parts[0], parts[1], parts[2] or "validation"
        print(f"[exp4] gold from HF dataset {name}/{conf} [{split}]")
        ds = load_dataset(name, conf, split=split)
        col = "text" if "text" in ds.column_names else ds.column_names[0]
        text = "\n".join(t for t in ds[col][:20000] if t)
        seqs = _chunk(tok(text)["input_ids"])
    else:
        print("[exp4] gold from OpenWebText via dataloader (needs --scratch-dir holding owt/ cache)")
        import dataloader
        _, valid = dataloader.get_dataloaders(cfg, tok, skip_train=True, valid_seed=cfg.seed)
        rows, got = [], 0
        for batch in valid:
            ids = batch["input_ids"][:, :L]
            if ids.size(1) < L:
                continue
            rows.append(ids); got += ids.size(0)
            if got >= n_seqs:
                break
        seqs = torch.cat(rows, 0)
    return seqs[:n_seqs].to(model.device)


def run_exp6(model, gold_tokens, steps, batch_size, seed, out):
    print(f"[exp6] gold reconstruction along sampler reveal schedule on {gold_tokens.size(0)} seqs ...")
    acc = {"reveal_step": [], "nll0": [], "nll_reveal": [], "top3_reveal": [], "top5_reveal": [], "H_traj": []}
    done, bi = 0, 0
    while done < gold_tokens.size(0):
        b = min(batch_size, gold_tokens.size(0) - done)
        o = H.gold_reconstruct_trace(model, gold_tokens[done:done + b], num_steps=steps, seed=seed + bi)
        for k in ("reveal_step", "nll0", "nll_reveal", "top3_reveal", "top5_reveal"):
            acc[k].append(o[k])
        acc["H_traj"].append(o["H_traj"])
        done += b; bi += 1
        print(f"  reconstructed {done}/{gold_tokens.size(0)}")
    data = {k: torch.cat(v, 0) for k, v in acc.items() if k != "H_traj"}
    data["H_traj"] = torch.cat(acc["H_traj"], dim=1)              # (S, N, L)
    data["num_steps"] = steps
    torch.save(data, os.path.join(out, "exp6_recon.pt"))
    print(f"  saved {out}/exp6_recon.pt  H_traj={tuple(data['H_traj'].shape)}")


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_ref(model, n, steps, batch_size, seed):
    """Self-generate ``n`` reference sequences, chunked by batch_size (avoids OOM)."""
    outs, done, bi = [], 0, 0
    while done < n:
        b = min(batch_size, n - done)
        outs.append(H.trace_sample(model, b, num_steps=steps, store_traj=False,
                                   seed=seed + bi)["final_tokens"])
        done += b; bi += 1
    return torch.cat(outs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all", help="comma list of {1,2,3,4,5} or 'all' (1&2 share a trace)")
    ap.add_argument("--ckpt", default="/data/imu-ml-security-project/Pretrained_Models/candi/candi-last.ckpt")
    ap.add_argument("--out", default="experiments/out/run1")
    ap.add_argument("--scratch-dir", default=os.environ.get("SCRATCH_DIR", "/home/patrick/.cache/discrete_diffusion"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-seqs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ratios", default="0,20,40,60,80")
    ap.add_argument("--diff-ratios", default="",
                    help="reveal %% at which to measure the difficulty score D_i for exp3 "
                         "(e.g. 10,20,30,40). Empty = use entropy at 0%% (legacy).")
    ap.add_argument("--perturb-ratios", default="20,40,60,80")
    ap.add_argument("--n-targets", type=int, default=8, help="exp3 target positions per ref sequence")
    ap.add_argument("--target-frac", type=float, default=0.2, help="exp4 fraction of positions kept masked")
    ap.add_argument("--frac", type=float, default=0.2, help="exp5 fraction of committed tokens perturbed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-ppl", action="store_true", help="exp5: compute GPT-2-large gen-PPL (downloads ~3GB)")
    ap.add_argument("--gold-file", default=None, help="exp4: gold from a UTF-8 text file")
    ap.add_argument("--gold-hf", default=None,
                    help="exp4: gold from HF dataset NAME[:CONFIG[:SPLIT]], e.g. "
                         "wikitext:wikitext-2-raw-v1:validation")
    args = ap.parse_args()

    ratios = [int(x) / 100.0 for x in args.ratios.split(",")]
    perturb_ratios = [int(x) / 100.0 for x in args.perturb_ratios.split(",")]
    diff_ratios = [int(x) / 100.0 for x in args.diff_ratios.split(",")] if args.diff_ratios else None
    exps = ["1", "2", "3", "4", "5", "6"] if args.exp == "all" else args.exp.split(",")
    os.makedirs(args.out, exist_ok=True)

    model, tok, cfg = load_model(args.ckpt, args.device, args.scratch_dir, length=1024)
    json.dump({"args": vars(args), "ratios": ratios, "perturb_ratios": perturb_ratios},
              open(os.path.join(args.out, "meta.json"), "w"), indent=2, default=str)

    ref_tokens = None  # cached self-generated sample for exp3

    if "1" in exps or "2" in exps:
        run_trace(model, args.n_seqs, args.steps, args.batch_size, args.seed, args.out)
        ref_tokens = torch.load(os.path.join(args.out, "exp1_2_trace.pt"))["final_tokens"]
        _free()

    if "3" in exps:
        n_ref = min(args.n_seqs, 8)
        if ref_tokens is None:
            ref_tokens = _generate_ref(model, n_ref, args.steps, args.batch_size, args.seed)
        run_exp3(model, ref_tokens[:n_ref], ratios, args.n_targets, args.batch_size, args.seed,
                 args.out, diff_ratios=diff_ratios)
        _free()

    if "4" in exps:
        try:
            gold = _load_gold(model, cfg, tok, args.n_seqs, args.gold_file, args.gold_hf)
            run_exp4(model, gold, ratios, args.target_frac, args.batch_size, args.seed, args.out)
        except Exception as e:
            print(f"[exp4] skipped: {e}\n"
                  f"       tip: pass --gold-hf wikitext:wikitext-2-raw-v1:validation "
                  f"(small download, no OWT cache needed) or --gold-file <some.txt>.")
        _free()

    if "5" in exps:
        run_exp5(model, tok, perturb_ratios, args.n_seqs, args.batch_size,
                 args.frac, args.seed, args.gen_ppl, args.out)
        _free()

    if "6" in exps:
        try:
            gold = _load_gold(model, cfg, tok, args.n_seqs, args.gold_file, args.gold_hf)
            run_exp6(model, gold, args.steps, args.batch_size, args.seed, args.out)
        except Exception as e:
            print(f"[exp6] skipped: {e}\n"
                  f"       tip: pass --gold-hf wikitext:wikitext-2-raw-v1:validation or --gold-file <txt>.")
        _free()

    print("done.")


if __name__ == "__main__":
    main()
