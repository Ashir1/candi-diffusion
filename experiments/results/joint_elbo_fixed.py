"""Corrected joint sequence-likelihood estimator for CANDI (Thread A unblock).

Replaces joint_elbo.py's Flow-GRPO surrogate continuous term with the VERIFIED
bridge-mixture log-likelihood (gate_e1 machinery, validated on the real
checkpoint to 1e-10 vs an independent implementation, ratio-level fix
validated to 5.8e-11 in round3/estimator_debug).

joint_logprob(model, x0, ...) returns the DIFFERENTIABLE joint pathwise
log-likelihood of a sequence under the model's current parameters:
  sum over steps of [reveal terms: log pi(x0_i | c)]   (discrete)
+ sum over steps/still-masked of [logsumexp mixture core]  (continuous)
Grid constants (lambda, lambdabar, Gaussian normalizers) are OMITTED by
default (constant_terms=False): the P1b constant-gap theorem certifies they
are policy-independent, so they cancel in GRPO ratios. Set
constant_terms=True to add the closed-form C for absolute-value use.

The corruption path (mask pattern + bridge residuals) is held FIXED via seed
so that ratios between two policies evaluate the same trajectory (CRN).
"""
import sys, math
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import torch
import torch.nn.functional as F
import numpy as np
import joint_elbo as je


def build_grid(model, n_steps, dev, t_hi=0.98, t_lo=0.02):
    taus = np.linspace(t_hi, t_lo, n_steps + 1)
    sig = model.get_continuous_from_discrete_noise(
        torch.tensor(taus, dtype=torch.float32, device=dev)).double().cpu().numpy()
    lam = [(taus[j-1]-taus[j])/taus[j-1] for j in range(1, n_steps+1)]
    lbar = [taus[j]/taus[j-1] for j in range(1, n_steps+1)]
    a = [(sig[j]**2)/(sig[j-1]**2) for j in range(1, n_steps+1)]
    v = [max(sig[j]**2*(1-sig[j]**2/sig[j-1]**2), 1e-12) for j in range(1, n_steps+1)]
    return taus, sig, lam, lbar, a, v


def grid_constant(grid, n_mask, V):
    """Closed-form C from the P1b constant-gap theorem."""
    taus, sig, lam, lbar, a, v = grid
    n = n_mask
    C = sum(math.log(lam[j-1]) for j in range(1, n+1))
    C += sum((n - j) * (math.log(lbar[j-1]) - (V/2)*math.log(2*math.pi*v[j-1]))
             for j in range(1, n))
    return C


def joint_logprob(model, x0, mask_pos, order, grid, seed, constant_terms=False):
    """Differentiable joint pathwise log-likelihood along a FIXED order/path.

    model: CANDI (backbone grads flow); x0: [L] token ids; mask_pos: list of
    masked positions; order: reveal order over mask_pos; grid: build_grid
    output; seed: fixes the bridge residuals (CRN across policies).
    """
    dev = x0.device
    V = je._real_vocab(model)
    taus, sig, lam, lbar, a, v = grid
    n = len(mask_pos)
    L = x0.shape[0]
    onehot = F.one_hot(x0, V).double()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    # pre-draw all bridge residuals (path fixed across policies)
    eps0 = {q: torch.randn(V, generator=gen, device=dev, dtype=torch.float64) for q in mask_pos}
    eps = {(q, j): torch.randn(V, generator=gen, device=dev, dtype=torch.float64)
           for q in mask_pos for j in range(1, n+1)}
    z = {q: onehot[q] + sig[0]*eps0[q] for q in mask_pos}
    revealed = set(range(L)) - set(mask_pos)
    still = list(mask_pos)
    total = torch.zeros((), device=dev)
    for j in range(1, n + 1):
        reveal = torch.zeros(L, device=dev)
        xt = torch.zeros(L, V, device=dev)
        for p in range(L):
            if p in revealed:
                reveal[p] = 1.0; xt[p] = onehot[p].float()
            else:
                xt[p] = z[p].float()
        logp = model.forward(xt=xt.unsqueeze(0),
                             discrete_noise=torch.tensor([float(taus[j-1])], device=dev),
                             reveal_mask=reveal.unsqueeze(0),
                             continuous_noise=torch.tensor([float(sig[j-1])], device=dev)
                             )[0].double()   # [L, V], DIFFERENTIABLE
        i_j = order[j-1]
        aj, vj = a[j-1], v[j-1]
        # discrete reveal term
        total = total + logp[i_j, x0[i_j]]
        # continuous mixture cores for still-masked (excluding the revealed one)
        for q in still:
            if q == i_j: continue
            zq = z[q]; e_y = onehot[q]
            z_new = e_y + aj*(zq - e_y) + math.sqrt(vj)*eps[(q, j)]
            w = z_new - aj*zq
            wn2 = (w*w).sum()
            scores = logp[q] - (wn2 - 2*(1-aj)*w + (1-aj)**2)/(2*vj)
            total = total + torch.logsumexp(scores, 0)
            z[q] = z_new
        still = [q for q in still if q != i_j]
        revealed.add(i_j)
    if constant_terms:
        total = total + grid_constant(grid, n, V)
    return total
