#!/usr/bin/env python
"""Analyze and plot results from the attention evolution experiment.

    python analyze_attention.py --data experiments/out/attn_exp/attn_evolution.pt \
        --out gen_imgs/attention_evolution/

Produces:
    - attention_heatmaps.png  — per-layer heatmaps of group→group attention over time
    - hard_hard_timeseries.png — how hard→hard attention evolves vs other pairs
    - attention_matrix_snapshots.png — group attention matrices at early/mid/late timesteps
"""
import argparse
import os

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


GROUP_LABELS = ["Easy (Q1)", "Med-Easy (Q2)", "Med-Hard (Q3)", "Hard (Q4)"]


def load_data(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def plot_timeseries(data, out_dir):
    """Plot attention from hard tokens to each group over time, per layer."""
    group_attn = data["group_attn"].numpy()  # (n_captured, n_layers, n_groups, n_groups)
    steps = data["captured_steps"]
    layers = data["layers"]
    n_groups = data["n_groups"]
    num_steps = data["num_steps"]

    # Normalize steps to fraction of denoising completed
    frac = [s / num_steps for s in steps]

    n_layers = len(layers)
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4), sharey=True)
    if n_layers == 1:
        axes = [axes]

    hard_idx = n_groups - 1  # hard tokens as queries

    for li, (ax, layer) in enumerate(zip(axes, layers)):
        for gj in range(n_groups):
            attn_vals = group_attn[:, li, hard_idx, gj]
            ax.plot(frac, attn_vals, label=f"Hard → {GROUP_LABELS[gj]}", linewidth=2)

        ax.set_xlabel("Denoising progress (fraction)")
        ax.set_title(f"Layer {layer}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean attention weight")
    fig.suptitle("Where do Hard tokens attend over time? (Gold reconstruction)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "hard_token_attention_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_all_pairs_timeseries(data, out_dir):
    """Plot all query-group → key-group pairs."""
    group_attn = data["group_attn"].numpy()
    steps = data["captured_steps"]
    layers = data["layers"]
    n_groups = data["n_groups"]
    num_steps = data["num_steps"]
    frac = [s / num_steps for s in steps]

    n_layers = len(layers)
    fig, axes = plt.subplots(n_groups, n_layers, figsize=(5 * n_layers, 3.5 * n_groups),
                             sharex=True, sharey=True)
    if n_layers == 1:
        axes = axes[:, None]

    for qi in range(n_groups):
        for li, layer in enumerate(layers):
            ax = axes[qi, li]
            for kj in range(n_groups):
                vals = group_attn[:, li, qi, kj]
                ax.plot(frac, vals, label=GROUP_LABELS[kj], linewidth=1.5)
            ax.grid(True, alpha=0.3)
            if qi == 0:
                ax.set_title(f"Layer {layer}")
            if li == 0:
                ax.set_ylabel(f"{GROUP_LABELS[qi]} →\nMean attn weight")
            if qi == n_groups - 1:
                ax.set_xlabel("Denoising progress")
            if qi == 0 and li == n_layers - 1:
                ax.legend(fontsize=7, title="Attending to:", loc="upper right")

    fig.suptitle("Group-to-Group Attention Evolution (Gold Reconstruction)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "all_pairs_attention_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_snapshots(data, out_dir):
    """Heatmap of the group attention matrix at early, mid, and late timesteps."""
    group_attn = data["group_attn"].numpy()
    steps = data["captured_steps"]
    layers = data["layers"]
    n_groups = data["n_groups"]
    num_steps = data["num_steps"]
    n_captured = len(steps)

    # Pick 3 snapshots: early (10%), mid (50%), late (90%)
    snap_fracs = [0.1, 0.5, 0.9]
    snap_indices = []
    for target in snap_fracs:
        target_step = int(target * num_steps)
        idx = min(range(n_captured), key=lambda i: abs(steps[i] - target_step))
        snap_indices.append(idx)

    n_layers = len(layers)
    fig, axes = plt.subplots(n_layers, 3, figsize=(4 * 3, 4 * n_layers))
    if n_layers == 1:
        axes = axes[None, :]

    for li, layer in enumerate(layers):
        for si, (snap_idx, frac) in enumerate(zip(snap_indices, snap_fracs)):
            ax = axes[li, si]
            mat = group_attn[snap_idx, li]  # (n_groups, n_groups)
            im = ax.imshow(mat, cmap="YlOrRd", vmin=0, aspect="equal")
            ax.set_xticks(range(n_groups))
            ax.set_xticklabels(["E", "ME", "MH", "H"], fontsize=9)
            ax.set_yticks(range(n_groups))
            ax.set_yticklabels(["E", "ME", "MH", "H"], fontsize=9)
            step_num = steps[snap_idx]
            ax.set_title(f"Step {step_num} ({int(frac*100)}%)", fontsize=10)
            if si == 0:
                ax.set_ylabel(f"Layer {layer}\n(Query group)")
            if li == n_layers - 1:
                ax.set_xlabel("Key group")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Annotate cells with values
            for gi in range(n_groups):
                for gj in range(n_groups):
                    ax.text(gj, gi, f"{mat[gi, gj]:.3f}", ha="center", va="center",
                            fontsize=8, color="black" if mat[gi, gj] < 0.4 else "white")

    fig.suptitle("Attention Between Difficulty Groups at Different Stages", fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "attention_matrix_snapshots.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_hard_hard_focus(data, out_dir):
    """Focused plot: Hard→Hard attention vs uniform baseline."""
    group_attn = data["group_attn"].numpy()
    steps = data["captured_steps"]
    layers = data["layers"]
    n_groups = data["n_groups"]
    num_steps = data["num_steps"]
    frac = [s / num_steps for s in steps]

    uniform_baseline = 1.0 / n_groups  # if attention were uniform across groups

    fig, ax = plt.subplots(figsize=(8, 5))
    for li, layer in enumerate(layers):
        hard_hard = group_attn[:, li, n_groups - 1, n_groups - 1]
        ax.plot(frac, hard_hard, linewidth=2, label=f"Layer {layer}")

    ax.axhline(uniform_baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Uniform baseline (1/{n_groups})")
    ax.set_yscale("log")
    ax.set_xlabel("Denoising progress (fraction of steps)", fontsize=11)
    ax.set_ylabel("Mean Hard→Hard attention weight (log scale)", fontsize=11)
    ax.set_title("Do Hard Tokens Preferentially Attend to Each Other?", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(out_dir, "hard_hard_attention_focus.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to attn_evolution.pt")
    parser.add_argument("--out", required=True, help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Loading data from {args.data} ...")
    data = load_data(args.data)

    print(f"  Captured steps: {len(data['captured_steps'])}")
    print(f"  Layers: {data['layers']}")
    print(f"  Groups: {data['n_groups']}")

    print("\nGenerating plots ...")
    plot_timeseries(data, args.out)
    plot_all_pairs_timeseries(data, args.out)
    plot_snapshots(data, args.out)
    plot_hard_hard_focus(data, args.out)

    print("\nAll plots saved!")


if __name__ == "__main__":
    main()
