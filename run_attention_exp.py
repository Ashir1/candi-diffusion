#!/usr/bin/env python
"""Experiment: How does attention between hard tokens evolve over diffusion timesteps?

Uses gold-text reconstruction (teacher-forced reveals) and hooks into the DDiTBlock
attention layers to track how hard-token ↔ hard-token attention changes as denoising
progresses.

    python run_attention_exp.py --out experiments/out/attn_exp \
        --n-seqs 8 --steps 256 --batch-size 2 \
        --gold-hf wikitext:wikitext-2-raw-v1:validation \
        --layers 0,5,11 --capture-every 16

Outputs:
    attn_evolution.pt  — per-timestep attention statistics between difficulty groups
    meta.json          — run metadata
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from denoise_diff import metrics as D
from denoise_diff.model import load_model, load_gold, free
from denoise_diff.attention_hooks import AttentionCapture


CKPT = os.path.join(os.path.dirname(__file__), "basecheckpoint", "candi-last.ckpt")


def _inference_schedule(model, num_steps, eps):
    """Replicate the schedule used by generate_samples_nocache."""
    timesteps = torch.linspace(0.999, eps, num_steps + 1, device=model.device)
    if model.use_percentile_scheduling:
        cont_noise = model.get_continuous_from_discrete_noise(timesteps)
    else:
        from algo import inference_sigmas
        cont_noise = inference_sigmas(num_steps + 1, model.sigma_min, model.sigma_max)
    dt = (1.0 - eps) / num_steps
    return timesteps, cont_noise, dt


@torch.no_grad()
def run_attention_evolution(model, gold, layers, capture_every=16,
                            num_steps=None, eps=1e-5, seed=42, n_groups=4):
    """Gold reconstruction with attention capture at regular intervals.

    Returns dict with:
        group_attn: (n_captured, n_layers, n_groups, n_groups)
            Mean attention from group_i -> group_j, averaged over heads and batch.
        group_attn_std: same shape, std across heads.
        captured_steps: list of step indices where attention was captured.
        H0: (B, L) initial entropy (difficulty scores).
        groups: (B, L) quantile group assignments (0=easy, n_groups-1=hard).
        reveal_step: (B, L) when each token was committed.
        timesteps_at_capture: list of t values at capture points.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if num_steps is None:
        num_steps = model.config.sampling.steps

    device = model.device
    gold = gold.to(device)
    B, L = gold.shape
    V = model.vocab_size - 1
    g = gold.clamp(max=V - 1)
    timesteps, cont_noise, _ = _inference_schedule(model, num_steps, eps)
    onehot = F.one_hot(g, V).float()

    cap = AttentionCapture(model.backbone.blocks, layers=layers)

    revealed = torch.zeros(B, L, device=device)
    reveal_step = torch.full((B, L), -1, dtype=torch.long, device=device)
    H0 = None

    # Storage for attention stats
    captured_steps = []
    timesteps_at_capture = []
    # group_attn[step_idx][layer_idx] = (n_groups, n_groups) mean attention
    all_group_attn = []
    all_group_attn_std = []

    for i in range(num_steps):
        t, s, sigma = timesteps[i], timesteps[i + 1], cont_noise[i]
        rm = revealed

        xt = (onehot * rm.unsqueeze(-1) +
              (1 - rm).unsqueeze(-1) * (onehot + sigma * torch.randn_like(onehot)))

        should_capture = (i % capture_every == 0) or (i == num_steps - 1)

        if should_capture:
            with cap.active():
                logp = model.forward(
                    xt=xt,
                    discrete_noise=torch.full((B,), float(t), device=device),
                    reveal_mask=rm,
                    continuous_noise=torch.full((B,), float(sigma), device=device),
                ).float()
        else:
            logp = model.forward(
                xt=xt,
                discrete_noise=torch.full((B,), float(t), device=device),
                reveal_mask=rm,
                continuous_noise=torch.full((B,), float(sigma), device=device),
            ).float()

        H = D.entropy_from_logprobs(logp)
        if i == 0:
            H0 = H.clone()

        # Commit tokens stochastically (teacher-forced to gold)
        prob = float(((t - s) / t).clamp(0, 1))
        newly = (revealed < 0.5) & (torch.rand(B, L, device=device) < prob)
        if newly.any():
            reveal_step[newly] = i
            revealed[newly] = 1.0

        # Process captured attention
        if should_capture:
            weights = cap.get_weights()
            h0_for_groups = H0.mean(0).cpu() if H0 is not None else H.mean(0).cpu()
            groups = D.quantile_groups(h0_for_groups, n_groups)  # (L,) on CPU

            step_attn_mean = []
            step_attn_std = []
            for layer_idx in layers:
                attn = weights[layer_idx]  # (B, n_heads, L, L)
                group_mat, group_std = _compute_group_attention(
                    attn, groups, n_groups
                )
                step_attn_mean.append(group_mat)
                step_attn_std.append(group_std)

            all_group_attn.append(torch.stack(step_attn_mean))  # (n_layers, n_groups, n_groups)
            all_group_attn_std.append(torch.stack(step_attn_std))
            captured_steps.append(i)
            timesteps_at_capture.append(float(t))
            cap.clear()

    # Handle never-revealed tokens
    never = reveal_step < 0
    reveal_step[never] = num_steps - 1

    # Recompute groups using H0 (per-sequence)
    final_groups = torch.stack([D.quantile_groups(H0[b], n_groups) for b in range(B)])

    return dict(
        group_attn=torch.stack(all_group_attn),        # (n_captured, n_layers, n_groups, n_groups)
        group_attn_std=torch.stack(all_group_attn_std),
        captured_steps=captured_steps,
        timesteps_at_capture=timesteps_at_capture,
        H0=H0.cpu(),
        groups=final_groups.cpu(),
        reveal_step=reveal_step.cpu(),
        num_steps=num_steps,
        n_groups=n_groups,
        layers=layers,
    )


def _compute_group_attention(attn, groups, n_groups):
    """Compute mean attention between difficulty groups.

    attn: (B, n_heads, L, L) — on CPU
    groups: (L,) int tensor

    Returns:
        group_mean: (n_groups, n_groups) — mean attn from group i (query) to group j (key)
        group_std: (n_groups, n_groups) — std across heads
    """
    B, H, L, _ = attn.shape
    attn_flat = attn.reshape(B * H, L, L)
    groups = groups.cpu()

    group_means = torch.zeros(B * H, n_groups, n_groups)
    for gi in range(n_groups):
        qi_mask = (groups == gi)
        if not qi_mask.any():
            continue
        for gj in range(n_groups):
            kj_mask = (groups == gj)
            if not kj_mask.any():
                continue
            sub = attn_flat[:, qi_mask][:, :, kj_mask]
            group_means[:, gi, gj] = sub.mean(dim=(-2, -1))

    return group_means.mean(0), group_means.std(0)


def main():
    parser = argparse.ArgumentParser(description="Attention evolution experiment")
    parser.add_argument("--ckpt", default=CKPT)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--n-seqs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--layers", type=str, default="0,5,11",
                        help="Comma-separated layer indices to capture")
    parser.add_argument("--capture-every", type=int, default=16,
                        help="Capture attention every N steps")
    parser.add_argument("--n-groups", type=int, default=4,
                        help="Number of difficulty quantile groups")
    parser.add_argument("--gold-hf", type=str, default=None)
    parser.add_argument("--gold-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    print(f"Loading model from {args.ckpt} ...")
    model, tok, cfg = load_model(args.ckpt, args.device)
    n_blocks = len(model.backbone.blocks)
    print(f"  Model has {n_blocks} blocks, capturing layers: {layers}")
    for l in layers:
        assert l < n_blocks, f"Layer {l} out of range (model has {n_blocks} blocks)"

    print(f"Loading gold text ({args.n_seqs} sequences) ...")
    gold = load_gold(model, cfg, tok, args.n_seqs,
                     gold_file=args.gold_file, gold_hf=args.gold_hf)
    print(f"  Gold shape: {gold.shape}")

    # Run in batches
    all_results = []
    done = 0
    while done < args.n_seqs:
        b = min(args.batch_size, args.n_seqs - done)
        batch_gold = gold[done:done + b]
        print(f"\nProcessing batch {done}-{done+b} ...")
        t0 = time.time()
        result = run_attention_evolution(
            model, batch_gold, layers=layers,
            capture_every=args.capture_every,
            num_steps=args.steps, seed=args.seed + done,
            n_groups=args.n_groups,
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        all_results.append(result)
        done += b
        free()

    # Aggregate across batches
    print("\nAggregating results ...")
    agg = {
        "group_attn": torch.stack([r["group_attn"] for r in all_results]).mean(0),
        "group_attn_std": torch.stack([r["group_attn_std"] for r in all_results]).mean(0),
        "captured_steps": all_results[0]["captured_steps"],
        "timesteps_at_capture": all_results[0]["timesteps_at_capture"],
        "H0": torch.cat([r["H0"] for r in all_results], dim=0),
        "groups": torch.cat([r["groups"] for r in all_results], dim=0),
        "reveal_step": torch.cat([r["reveal_step"] for r in all_results], dim=0),
        "num_steps": args.steps,
        "n_groups": args.n_groups,
        "layers": layers,
    }

    out_path = os.path.join(args.out, "attn_evolution.pt")
    torch.save(agg, out_path)
    print(f"Saved: {out_path}")

    meta = dict(
        ckpt=args.ckpt, n_seqs=args.n_seqs, steps=args.steps,
        batch_size=args.batch_size, layers=layers,
        capture_every=args.capture_every, n_groups=args.n_groups,
        seed=args.seed, gold_hf=args.gold_hf, gold_file=args.gold_file,
        device=args.device, n_captured=len(agg["captured_steps"]),
    )
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved: {args.out}/meta.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
