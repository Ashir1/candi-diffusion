"""Read-only instrumentation harnesses for the denoise-vs-difficulty study.

These functions drive the **existing, unmodified** CANDI model. They re-implement
the non-cached sampling loop (``algo.CANDI.generate_samples_nocache``) but log
per-step diagnostics, and add a single-forward "probe" and a "perturb-and-continue"
intervention. Nothing in ``algo.py`` / the model is changed.

Harnesses
---------
* ``trace_sample``        -- H1: full sampler trajectory (entropy field, reveal steps). exp 1, exp 2.
* ``denoiser_probe``      -- H2: one forward with a constructed reveal set.          exp 3 (ref-free), exp 4 (gold).
* ``perturb_and_continue``-- exp 5: replace committed tokens with wrong tokens, continue sampling.
"""

import torch
import torch.nn.functional as F

import difficulty as D


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
# exp 5 -- perturb committed tokens, then continue the unmodified sampler
# --------------------------------------------------------------------------- #
@torch.no_grad()
def perturb_and_continue(model, num_samples, ratio_r, selection=None, frac=0.2,
                         num_steps=None, eps=1e-5, seed=0):
    """Run the sampler; when the global clean ratio first reaches ``ratio_r``,
    replace ``frac`` of the currently-committed tokens (selected by initial
    difficulty) with a *wrong* token, then continue to completion.

    selection : None (baseline, no perturbation) | 'hard' | 'random' | 'easy'.
    Same ``seed`` across conditions makes the perturbation the only exogenous change.

    Returns CPU tensors: final_tokens (B,L), perturb_mask (B,L) bool.
    """
    set_seed(seed)
    if num_steps is None:
        num_steps = model.config.sampling.steps
    B, L = num_samples, model.num_tokens
    device = model.device

    x = model.prior_sample(B, L)
    clean = torch.zeros((B, L), device=device)
    timesteps, cont_noise, dt = _inference_schedule(model, num_steps, eps)
    model.max_sigma = cont_noise.max().item()
    V = x.size(-1)

    H0 = None
    perturb_mask = torch.zeros((B, L), dtype=torch.bool, device=device)
    done = False

    for i in range(num_steps):
        t, s = timesteps[i], timesteps[i + 1]
        x_cont, p_x0 = model._continuous_step(
            x, t, sigma_s=cont_noise[i], sigma_t=cont_noise[i + 1],
            clean_mask=clean, time_s=s)
        if i == 0:
            H0 = D.predictive_entropy(p_x0.float())
        x, clean = model._discrete_step(x_cont, p_x0, t, dt, prev_clean_mask=clean)

        if (not done) and selection is not None and clean.float().mean().item() >= ratio_r:
            ids = x.argmax(dim=-1)
            for b in range(B):
                idx = clean[b].bool().nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                scores = H0[b, idx]
                kk = max(1, int(round(frac * idx.numel())))
                if selection == "hard":
                    sel = idx[scores.topk(min(kk, idx.numel())).indices]
                elif selection == "easy":
                    sel = idx[(-scores).topk(min(kk, idx.numel())).indices]
                else:  # random
                    perm = torch.randperm(idx.numel(), device=device)[:kk]
                    sel = idx[perm]
                cur = ids[b, sel]
                rnd = torch.randint(0, V, (sel.numel(),), device=device)
                rnd = torch.where(rnd == cur, (rnd + 1) % V, rnd)          # ensure "wrong"
                x[b, sel] = F.one_hot(rnd, V).to(x.dtype)
                perturb_mask[b, sel] = True
            done = True

    return dict(final_tokens=x.argmax(dim=-1).cpu(), perturb_mask=perturb_mask.cpu())


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
