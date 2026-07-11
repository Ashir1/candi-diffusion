"""Coupling measurement v2: work in reduced dimensions.

The coupling residual for GRPO is really about the IMPORTANCE RATIO error.
For a pair (theta_new, theta_old) with posteriors pi_new and pi_old:
  log r_joint = log(sum_y pi_new(y) phi_y(x_s)) - log(sum_y pi_old(y) phi_y(x_s))
  log r_factored = log pi_new(k) - log pi_old(k)   [for the revealed token k]
                 + log N(x_s; mu_new, v) - log N(x_s; mu_old, v)  [factored continuous]

The difference is: log r_joint - log r_factored = coupling_new - coupling_old
where coupling = log M(x_s) - log pi(k) - log N(x_s; E_pi[mu], v)

We can measure this on the REAL model by perturbing logits (simulating a policy update)
and comparing the joint ratio to the factored ratio.
"""
import sys, os, json, math, time
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import joint_elbo as je
import torch, torch.nn.functional as F
import numpy as np

print("Loading model...")
model, tok, cfg, load_info = je.load_real_model(device="cuda", length=None)
V = je._real_vocab(model)
print(f"Model loaded: V={V}")

torch.manual_seed(123)
B, L = 2, 128
x0 = torch.randint(0, V, (B, L), device="cuda")
onehot = F.one_hot(x0, V).float()

results = {}
for t_val in [0.3, 0.5, 0.7, 0.9]:
    t_vec = torch.full((B,), t_val, device="cuda")
    dalpha, alpha = model.noise(t_vec)
    disc_noise = (1 - alpha).float()
    sigma_val = model.get_continuous_from_discrete_noise(disc_noise).reshape(B).float()
    
    # Construct xt
    move = torch.rand(B, L, device="cuda") < (1 - alpha).view(B, 1)
    random_tok = torch.randint(0, V, (B, L), device="cuda")
    xt_tok = torch.where(move, random_tok, x0)
    reveal = (xt_tok == x0).float()
    noise_eps = torch.randn(B, L, V, device="cuda")
    xt_cont = onehot + sigma_val[:, None, None] * noise_eps
    xt = onehot * reveal[:,:,None] + (1 - reveal)[:,:,None] * xt_cont
    
    # Get model posterior (old policy)
    with torch.no_grad():
        logp_old = model.forward(xt=xt, discrete_noise=disc_noise, reveal_mask=reveal, continuous_noise=sigma_val)
    
    # Simulate new policy: add small perturbation to logits
    logp_new = logp_old + 0.1 * torch.randn_like(logp_old)
    logp_new = logp_new - logp_new.logsumexp(-1, keepdim=True)  # re-normalize
    
    pi_old = logp_old.exp()
    pi_new = logp_new.exp()
    
    # For masked positions, compute joint ratio vs factored ratio
    masked = (reveal == 0)
    if not masked.any():
        continue
    
    # We evaluate the ratio at the ground truth token x0
    # Joint ratio for REVEAL branch: pi_new(x0) / pi_old(x0)
    # This is the same as factored discrete for the reveal branch
    
    # For the MASKED branch (stays masked), the joint observation x_s is the continuous state
    # Joint: M_new(x_s|x_t) / M_old(x_s|x_t)
    # where M(x_s|x_t) = sum_y pi(y) * phi_y(x_s|x_t)
    # Factored: [pi_new(mode) / pi_old(mode)] * [N(x_s; mu_new, v) / N(x_s; mu_old, v)]
    
    # The key insight: in the high-dimensional space, we should work with 
    # log-ratio differences, which cancel the normalizing constants.
    # log M_theta(x_s|x_t) = log sum_y pi(y) exp(-||x_s - mu_y||^2 / (2v))
    #                        - (V/2) log(2*pi*v)
    # The normalization cancels in the ratio, so:
    # log(M_new/M_old) = logsumexp_y(log pi_new(y) - ||x_s-mu_y||^2/(2v))
    #                   - logsumexp_y(log pi_old(y) - ||x_s-mu_y||^2/(2v))
    
    # For computational tractability, use only top-K tokens
    topK = 64
    
    # Sample some masked positions
    masked_flat = masked.nonzero()  # [N, 2]
    N_sample = min(masked_flat.shape[0], 100)
    perm = torch.randperm(masked_flat.shape[0], device="cuda")[:N_sample]
    sampled = masked_flat[perm]
    
    joint_log_ratios = []
    factored_log_ratios = []
    
    for idx in range(N_sample):
        b_i, l_i = sampled[idx, 0].item(), sampled[idx, 1].item()
        sigma_i = sigma_val[b_i].item()
        
        # Get x_s (the continuous state at this masked position)
        x_s = xt[b_i, l_i]  # [V]
        
        # Get posteriors
        lp_old_i = logp_old[b_i, l_i]  # [V]
        lp_new_i = logp_new[b_i, l_i]  # [V]
        
        # Top-K tokens for mixture
        topk_old_vals, topk_old_idx = lp_old_i.topk(topK)
        topk_new_vals, topk_new_idx = lp_new_i.topk(topK)
        # Union of both top-K sets
        all_idx = torch.unique(torch.cat([topk_old_idx, topk_new_idx]))
        K_eff = all_idx.shape[0]
        
        # For the Gaussian bridge phi_y(x_s|x_t):
        # mu_y = (1-a)*e_y + a*x_t  where x_t is x_s here (since we evaluate at the current state)
        # Actually for the masked branch at step (s,t):
        # The bridge mu_y(x_t) = e_y + a*(x_t - e_y) where a = sigma_s^2/sigma_t^2
        # For ratio purposes the quadratic form is:
        # -||x_s - mu_y||^2 / (2v) = -||x_s - e_y - a*(x_t - e_y)||^2 / (2v)
        # = -||(1-a)(x_s - e_y) + ... no, we need to keep the exact expr
        
        # Actually for a self-consistent ratio, we just need:
        # log sum_y pi(y) * exp(-||x_s - mu_y||^2/(2v))
        # where mu_y = (1-a)*e_y + a*x_t_original
        # Since the ratio cancels -(V/2)log(2pi*v), we compute unnormalized
        
        # Let's use a simpler and more numerically stable approach.
        # The coupling that matters for GRPO is:
        # ratio_joint = sum_y pi_new(y) * w_y / sum_y pi_old(y) * w_y
        # where w_y = phi_y(x_s|x_t) (the bridge density at x_s)
        # ratio_factored = (sum_y pi_new(y) * w_y / sum_y pi_new(y)) /
        #                  (sum_y pi_old(y) * w_y / sum_y pi_old(y))
        # ... actually that's exactly the joint ratio again.
        
        # The factored ratio is: pi_new(mode)/pi_old(mode) for discrete
        # times something for continuous. But there is no separate "continuous ratio"
        # in the masked branch because the continuous observation IS the mixture.
        
        # The correct comparison is:
        # JOINT log ratio = log(sum_y pi_new(y)*w_y) - log(sum_y pi_old(y)*w_y)
        # FACTORED log ratio = log(pi_new(y*)) - log(pi_old(y*))
        #   where y* = argmax pi_old(y) (the mode, treating as if discrete only)
        
        # Compute w_y for each token in all_idx
        # w_y proportional to exp(-||x_s - mu_y||^2 / (2v))
        # But in V=50258 dims this is numerically challenging.
        # Key: mu_y differs from mu_y' only in ONE coordinate (the y-th and y'-th).
        # mu_y = a*x_t + (1-a)*e_y
        # So ||x_s - mu_y||^2 = ||x_s - a*x_t||^2 + (1-a)^2 - 2*(1-a)*(x_s - a*x_t)_y
        # where (.)_y means the y-th component.
        # The first term is constant across y! So:
        # -||x_s - mu_y||^2/(2v) = const - (1-a)^2/(2v) + (1-a)*(x_s - a*x_t)_y / v
        # = const + (1-a)/v * [(x_s - a*x_t)_y - (1-a)/2]
        
        # This means: log w_y = const + (1-a)/v * (x_s_y - a*x_t_y - (1-a)/2)
        # ... wait, let me be more careful
        # mu_y[j] = a*x_t[j] + (1-a)*delta(j,y)
        # x_s[j] - mu_y[j] = (x_s[j] - a*x_t[j]) - (1-a)*delta(j,y)
        # ||x_s - mu_y||^2 = sum_j (x_s[j]-a*x_t[j])^2 - 2*(1-a)*(x_s[y]-a*x_t[y]) + (1-a)^2
        # = C - 2*(1-a)*(x_s[y] - a*x_t[y]) + (1-a)^2
        # where C = ||x_s - a*x_t||^2 is constant over y.
        
        # So: -||x_s-mu_y||^2/(2v) = -C/(2v) + (1-a)*(x_s[y]-a*x_t[y])/v - (1-a)^2/(2v)
        # log w_y = (1-a)/v * (x_s[y] - a*x_t[y]) + const_over_y
        
        # This is MUCH better numerically! We just need (x_s[y] - a*x_t[y]) for each y.
        
        # Now let's compute a proper coupling measurement.
        # For masked branch, assume dt=0.1, so s=t-dt
        dt = 0.1
        s = t_val - dt
        sigma_s = model.get_continuous_from_discrete_noise(
            torch.tensor([1 - (1-0.001)*s], device="cuda")).float().item()
        sigma_t_i = sigma_i
        a = sigma_s**2 / (sigma_t_i**2 + 1e-12)
        v = sigma_s**2 * max(1 - sigma_s**2/(sigma_t_i**2+1e-12), 1e-12)
        
        # x_t for this position
        x_t_pos = xt[b_i, l_i]  # [V]
        
        # log w_y = (1-a)/v * (x_s[y] - a*x_t[y])  (up to a constant over y)
        # x_s = x_t for the stay-masked branch (we evaluate at x_s = current state)
        # Actually x_s is not x_t; x_s is the continuous state at the NEXT (cleaner) step.
        # For the ratio evaluation, we sample x_s from the bridge.
        # Let's use x_s = x_t (a common evaluation point for the density)
        x_s_eval = x_t_pos
        
        coeff = (1 - a) / max(v, 1e-12)
        # log_w_y (unnormalized) for all y in union set:
        # We need x_s[y] - a*x_t[y] for each y in all_idx
        xs_minus_axt = x_s_eval - a * x_t_pos  # [V]
        log_w_unnorm = coeff * xs_minus_axt[all_idx]  # [K_eff]
        
        # log pi values
        lp_old_sel = lp_old_i[all_idx]  # [K_eff]
        lp_new_sel = lp_new_i[all_idx]  # [K_eff]
        
        # Joint log ratio:
        # log(sum pi_new(y)*w_y) - log(sum pi_old(y)*w_y)
        # = logsumexp(log_pi_new + log_w) - logsumexp(log_pi_old + log_w)
        log_joint_new = torch.logsumexp(lp_new_sel + log_w_unnorm, 0).item()
        log_joint_old = torch.logsumexp(lp_old_sel + log_w_unnorm, 0).item()
        log_ratio_joint = log_joint_new - log_joint_old
        
        # Factored log ratio (discrete only, at the MAP token):
        mode_old = all_idx[lp_old_sel.argmax()].item()
        log_ratio_factored = lp_new_i[mode_old].item() - lp_old_i[mode_old].item()
        
        joint_log_ratios.append(log_ratio_joint)
        factored_log_ratios.append(log_ratio_factored)
    
    joint_arr = torch.tensor(joint_log_ratios)
    fact_arr = torch.tensor(factored_log_ratios)
    error = joint_arr - fact_arr
    
    results[f"t={t_val}"] = {
        "n_samples": N_sample,
        "joint_ratio_mean": float(joint_arr.mean()),
        "factored_ratio_mean": float(fact_arr.mean()),
        "error_mean": float(error.mean()),
        "error_std": float(error.std()),
        "error_abs_mean": float(error.abs().mean()),
        "error_abs_max": float(error.abs().max()),
        "error_abs_p90": float(error.abs().quantile(0.9)),
        "error_abs_p99": float(error.abs().quantile(0.99) if N_sample >= 100 else error.abs().max()),
        "correlation": float(torch.corrcoef(torch.stack([joint_arr, fact_arr]))[0,1]) if N_sample > 2 else 0.0,
    }
    print(f"t={t_val}: error mean={error.mean():.4f} std={error.std():.4f} abs_max={error.abs().max():.4f} corr={results[f't={t_val}']['correlation']:.4f}")

out_path = "/home/ec2-user/together/shared/artifacts/coupling_real_checkpoint.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
print(json.dumps(results, indent=2))
print("DONE")
