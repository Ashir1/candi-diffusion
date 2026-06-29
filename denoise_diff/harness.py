"""Read-only instrumentation harnesses for the denoise-vs-difficulty study.

These functions drive the **existing, unmodified** CANDI model. They re-implement
the non-cached sampling loop (``algo.CANDI.generate_samples_nocache``) but log
per-step diagnostics, and add a single-forward "probe" and a "perturb-and-continue"
intervention. Nothing in ``algo.py`` / the model is changed.

Harnesses
---------
* ``trace_sample``           -- full sampler trajectory (entropy field, reveal steps).  exp 1, exp 2.
* ``denoiser_probe``         -- one forward with a constructed reveal set.              exp 3, exp 4.
* ``gold_reconstruct_trace`` -- reconstruct real text along the reveal schedule.        exp 6.
* ``gold_renoise_trace``     -- gold reconstruction + re-noise intervention.            exp 7.
"""

import torch
import torch.nn.functional as F

from . import metrics as D


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _inference_schedule(model, num_steps, eps):
    """Replicate the schedule used by ``generate_samples_nocache``."""
    timesteps = torch.linspace(0.999, eps, num_steps + 1, device=model.device)
    if model.use_percentile_scheduling:
        cont_noise = model.get_continuous_from_discrete_noise(timesteps)
    else:
        from algo import inference_sigmas
        cont_noise = inference_sigmas(num_steps + 1, model.sigma_min, model.sigma_max)
    dt = (1.0 - eps) / num_steps
    return timesteps, cont_noise, dt


# --------------------------------------------------------------------------- #
# H1 -- trajectory logging
# --------------------------------------------------------------------------- #
@torch.no_grad()
def trace_sample(model, num_samples, num_steps=None, eps=1e-5,
                 store_traj=True, seed=None):
    """Run the unmodified non-cached sampler and log per-step diagnostics.

    Returns CPU tensors:
      final_tokens (B,L) long, reveal_step (B,L) long, committed (B,L) long,
      H0 (B,L) float (entropy at step 0 = initial difficulty D_i),
      [H_traj (S,B,L) fp16 if store_traj].
    """
    if seed is not None:
        set_seed(seed)
    if num_steps is None:
        num_steps = model.config.sampling.steps
    B, L = num_samples, model.num_tokens
    device = model.device

    x = model.prior_sample(B, L)                                   # (B,L,V)
    clean = torch.zeros((B, L), device=device)
    timesteps, cont_noise, dt = _inference_schedule(model, num_steps, eps)
    model.max_sigma = cont_noise.max().item()

    reveal_step = torch.full((B, L), -1, dtype=torch.long, device=device)
    committed = torch.zeros((B, L), dtype=torch.long, device=device)
    H0 = None
    H_traj = [] if store_traj else None

    for i in range(num_steps):
        t, s = timesteps[i], timesteps[i + 1]
        x_cont, p_x0 = model._continuous_step(
            x, t, sigma_s=cont_noise[i], sigma_t=cont_noise[i + 1],
            clean_mask=clean, time_s=s)
        H = D.predictive_entropy(p_x0.float())                     # (B,L) nats, pre-commit
        if i == 0:
            H0 = H.clone()
        if store_traj:
            H_traj.append(H.half().cpu())
        old_clean = clean.bool().clone()
        x, clean = model._discrete_step(x_cont, p_x0, t, dt, prev_clean_mask=clean)
        newly = clean.bool() & ~old_clean
        if newly.any():
            ids = x.argmax(dim=-1)
            reveal_step[newly] = i
            committed[newly] = ids[newly]

    final = x.argmax(dim=-1)
    never = reveal_step < 0
    reveal_step[never] = num_steps - 1
    committed[never] = final[never]

    out = dict(final_tokens=final.cpu(), reveal_step=reveal_step.cpu(),
               committed=committed.cpu(), H0=H0.cpu(),
               num_steps=int(num_steps), L=int(L))
    if store_traj:
        out["H_traj"] = torch.stack(H_traj, dim=0)                 # (S,B,L) fp16 cpu
    return out


# --------------------------------------------------------------------------- #
# H2 -- single-forward probe at a fixed reveal ratio
# --------------------------------------------------------------------------- #
@torch.no_grad()
def denoiser_probe(model, x_ref, reveal_mask, ratio, gold=None):
    """One denoiser forward with a *chosen* reveal set.

    x_ref       : (B,L) token ids (gold for exp 4, or a self-generated sample for exp 3).
    reveal_mask : (B,L) bool/float, True = revealed (clean context).
    ratio       : clean fraction r in [0,1] -> sets the noise level (disc = 1-r).
    gold        : (B,L) ids; if given, also returns gold-NLL / accuracy (exp 4).

    Returns CPU tensors: entropy (B,L), masked (B,L) bool, [gold_nll, correct, top5].
    """
    device = model.device
    x_ref = x_ref.to(device)
    rm = reveal_mask.to(device).float()
    B, L = x_ref.shape
    V = model.vocab_size - 1                                       # real vocab (no mask col)

    disc = min(max(1.0 - float(ratio), 1e-5), 0.999)               # avoid disc==1 (sigma_map -> inf)
    sigma = model.get_continuous_from_discrete_noise(
        torch.tensor([disc], device=device)).reshape(-1)[0]

    onehot = F.one_hot(x_ref.clamp(max=V - 1), num_classes=V).float()
    xt_cont = onehot + sigma * torch.randn_like(onehot)
    xt = onehot * rm.unsqueeze(-1) + (1.0 - rm).unsqueeze(-1) * xt_cont

    disc_vec = torch.full((B,), disc, device=device)
    sigma_vec = torch.full((B,), float(sigma), device=device)
    logp = model.forward(xt=xt, discrete_noise=disc_vec,
                         reveal_mask=rm, continuous_noise=sigma_vec).float()   # (B,L,V)

    out = dict(entropy=D.entropy_from_logprobs(logp).cpu(),
               masked=(rm < 0.5).cpu())
    if gold is not None:
        g = gold.to(device).clamp(max=V - 1)
        out["gold_nll"] = D.gold_nll(logp, g).cpu()
        out["correct"] = D.argmax_accuracy(logp, g).cpu()        # top-1
        out["top3"] = D.topk_accuracy(logp, g, 3).cpu()
        out["top5"] = D.topk_accuracy(logp, g, 5).cpu()
    return out


# --------------------------------------------------------------------------- #
# exp 6 -- rescue along the sampler's stochastic reveal schedule, on real text
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gold_reconstruct_trace(model, gold, num_steps=None, eps=1e-5, seed=None):
    """Reconstruct real text ``gold`` by running the sampler's random reveal schedule with
    teacher-forced gold reveals. Each position gets a random reveal time tau; we score the
    model's prediction (vs gold) at tau, the moment it has accumulated the most context.

    Returns CPU tensors:
      H_traj (S,B,L) fp16   -- entropy field (for difficulty-at-rho and early entropy),
      reveal_step (B,L),
      nll0, nll_reveal (B,L)        -- gold-NLL at step 0 vs at reveal time,
      top3_reveal, top5_reveal (B,L)-- top-k accuracy (vs gold) at reveal time.
    """
    if seed is not None:
        set_seed(seed)
    if num_steps is None:
        num_steps = model.config.sampling.steps
    device = model.device
    gold = gold.to(device)
    B, L = gold.shape
    V = model.vocab_size - 1
    g = gold.clamp(max=V - 1)
    timesteps, cont_noise, _ = _inference_schedule(model, num_steps, eps)
    onehot = F.one_hot(g, V).float()

    revealed = torch.zeros(B, L, device=device)
    reveal_step = torch.full((B, L), -1, dtype=torch.long, device=device)
    nll0 = None
    nll_rev = torch.zeros(B, L, device=device)
    t3_rev = torch.zeros(B, L, device=device)
    t5_rev = torch.zeros(B, L, device=device)
    H_traj = []

    for i in range(num_steps):
        t, s, sigma = timesteps[i], timesteps[i + 1], cont_noise[i]
        rm = revealed
        xt = onehot * rm.unsqueeze(-1) + (1 - rm).unsqueeze(-1) * (onehot + sigma * torch.randn_like(onehot))
        logp = model.forward(xt=xt, discrete_noise=torch.full((B,), float(t), device=device),
                             reveal_mask=rm, continuous_noise=torch.full((B,), float(sigma), device=device)).float()
        H = D.entropy_from_logprobs(logp)
        NLL = D.gold_nll(logp, g)
        H_traj.append(H.half().cpu())
        if i == 0:
            nll0 = NLL.clone()
        prob = float(((t - s) / t).clamp(0, 1))
        newly = (revealed < 0.5) & (torch.rand(B, L, device=device) < prob)
        if newly.any():
            reveal_step[newly] = i
            nll_rev[newly] = NLL[newly]
            t3_rev[newly] = D.topk_accuracy(logp, g, 3)[newly]
            t5_rev[newly] = D.topk_accuracy(logp, g, 5)[newly]
            revealed[newly] = 1.0

    never = reveal_step < 0                                   # never committed -> use final step
    if never.any():
        reveal_step[never] = num_steps - 1
        nll_rev[never] = NLL[never]
        t3_rev[never] = D.topk_accuracy(logp, g, 3)[never]
        t5_rev[never] = D.topk_accuracy(logp, g, 5)[never]
    return dict(H_traj=torch.stack(H_traj, 0), reveal_step=reveal_step.cpu(),
                nll0=nll0.cpu(), nll_reveal=nll_rev.cpu(),
                top3_reveal=t3_rev.cpu(), top5_reveal=t5_rev.cpu(), num_steps=int(num_steps))


# --------------------------------------------------------------------------- #
# exp 7 -- gold reconstruction + re-noise intervention, with full trajectories
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gold_renoise_trace(model, gold, intervention_ratio, selection, frac=0.2,
                       num_steps=None, eps=1e-5, seed=None):
    """Reconstruct real ``gold`` via the random reveal schedule (teacher-forced). At
    ``intervention_ratio`` un-commit a hard/random/easy subset of the revealed tokens
    (ranked by entropy-at-reveal) and let the model RE-SAMPLE them (free choice) as
    denoising continues. Tracks per-step trajectories over the still-masked tokens.

    selection: None (baseline, no re-noise) | 'hard' | 'random' | 'easy'.
    Returns per-step (num_steps,) trajectories + final tokens (gold except re-noised = model's choice).
    """
    if seed is not None:
        set_seed(seed)
    if num_steps is None:
        num_steps = model.config.sampling.steps
    device = model.device
    gold = gold.to(device)
    B, L = gold.shape
    V = model.vocab_size - 1
    g = gold.clamp(max=V - 1)
    timesteps, cont_noise, _ = _inference_schedule(model, num_steps, eps)
    gold_oh = F.one_hot(g, V).float()

    committed = torch.zeros(B, L, device=device)
    comm_val = g.clone()
    renoise_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    ent_at_reveal = torch.zeros(B, L, device=device)
    revealed_once = torch.zeros(B, L, dtype=torch.bool, device=device)
    intervened = (selection is None)
    keys = ["denoise_ratio", "ent_all", "ent_renoise", "ent_non",
            "acc3_all", "acc3_renoise", "acc3_non", "acc5_all", "acc5_renoise", "acc5_non"]
    traj = {k: [] for k in keys}

    def _m(t, mask):
        return t[mask].mean().item() if mask.any() else float("nan")

    for i in range(num_steps):
        t, s, sigma = timesteps[i], timesteps[i + 1], cont_noise[i]
        clean = committed
        comm_oh = F.one_hot(comm_val, V).float()
        xt = comm_oh * clean.unsqueeze(-1) + (1 - clean).unsqueeze(-1) * (gold_oh + sigma * torch.randn_like(gold_oh))
        logp = model.forward(xt=xt, discrete_noise=torch.full((B,), float(t), device=device),
                             reveal_mask=clean, continuous_noise=torch.full((B,), float(sigma), device=device)).float()
        H = D.entropy_from_logprobs(logp)
        t3 = D.topk_accuracy(logp, g, 3); t5 = D.topk_accuracy(logp, g, 5)
        masked = committed < 0.5
        mr = masked & renoise_mask; mn = masked & ~renoise_mask
        traj["denoise_ratio"].append(committed.float().mean().item())
        traj["ent_all"].append(_m(H, masked)); traj["ent_renoise"].append(_m(H, mr)); traj["ent_non"].append(_m(H, mn))
        traj["acc3_all"].append(_m(t3, masked)); traj["acc3_renoise"].append(_m(t3, mr)); traj["acc3_non"].append(_m(t3, mn))
        traj["acc5_all"].append(_m(t5, masked)); traj["acc5_renoise"].append(_m(t5, mr)); traj["acc5_non"].append(_m(t5, mn))

        prob = float(((t - s) / t).clamp(0, 1))
        newly = (committed < 0.5) & (torch.rand(B, L, device=device) < prob)
        first = newly & ~revealed_once
        ent_at_reveal[first] = H[first]; revealed_once |= first
        if newly.any():
            sample = logp.argmax(-1)                          # re-noised tokens commit to model's choice
            comm_val[newly] = torch.where(renoise_mask[newly], sample[newly], g[newly])
            committed[newly] = 1.0

        if (not intervened) and committed.float().mean().item() >= intervention_ratio:
            for b in range(B):
                idx = (committed[b] > 0.5).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                scores = ent_at_reveal[b, idx]
                kk = max(1, int(round(frac * idx.numel())))
                if selection == "hard":
                    sel = idx[scores.topk(min(kk, idx.numel())).indices]
                elif selection == "easy":
                    sel = idx[(-scores).topk(min(kk, idx.numel())).indices]
                else:
                    sel = idx[torch.randperm(idx.numel(), device=device)[:kk]]
                committed[b, sel] = 0.0; renoise_mask[b, sel] = True
            intervened = True

    out = {k: torch.tensor(traj[k]) for k in keys}
    out["final_tokens"] = comm_val.cpu()
    out["renoise_mask"] = renoise_mask.cpu()
    out["num_steps"] = int(num_steps)
    return out
