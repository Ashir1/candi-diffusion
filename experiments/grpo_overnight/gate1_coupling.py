import sys, os, time, json, math
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import joint_elbo as je
import torch, torch.nn.functional as F
import numpy as np

model, tok, cfg, load_info = je.load_real_model(device="cuda", length=None)
torch.manual_seed(20260710)
x_real = je._make_batch(tok, 64, "cuda")
V = je._real_vocab(model)
z_real = F.one_hot(x_real.clamp(0, V-1), num_classes=V).float()
cond = {"seed": 20260710, "num_steps": 8, "continuous_eps": 0.05, "transition_scale": 0.75}

result = je.joint_elbo_with_parts(model, x_real, z_real, cond, n_quad=3, k_mc=1)
print(f"Total={result['joint_elbo']:.1f}, Disc={result['discrete_elbo']:.4f}, Cont={result['continuous_logprob']:.1f}")

# Shuffled contrast
g = torch.Generator(device="cuda"); g.manual_seed(20260710+11)
idx = torch.randperm(64, generator=g, device="cuda")
x_shuf = x_real[:, idx]
z_shuf = F.one_hot(x_shuf.clamp(0, V-1), num_classes=V).float()
shuf_result = je.joint_elbo_with_parts(model, x_shuf, z_shuf, cond, n_quad=3, k_mc=1)
print(f"Shuffled={shuf_result['joint_elbo']:.1f}")
delta = result["joint_elbo"] - shuf_result["joint_elbo"]
print(f"Contrast: {delta:.1f}, Gate 1 basic: {delta > 0}")

# Differentiable forward test
print("\n=== Differentiable forward path ===")
model.train()
for p in model.parameters():
    p.requires_grad_(True)
t_vec = torch.full((1,), 0.5, device="cuda")
dalpha, alpha = model.noise(t_vec)
disc_noise = (1.0 - alpha).float()
sigma = model.get_continuous_from_discrete_noise(disc_noise).reshape(1).float()
onehot = F.one_hot(x_real.clamp(0, V-1), num_classes=V).float()
noise_eps = torch.randn_like(onehot) * sigma
xt = onehot + noise_eps
reveal = torch.zeros(1, 64, device="cuda", dtype=torch.float32)
logp = model.forward(xt=xt, discrete_noise=disc_noise.expand(1), reveal_mask=reveal, continuous_noise=sigma.expand(1))
print(f"  logp shape: {logp.shape}, grad: {logp.requires_grad}")
target_logp = logp.gather(-1, x_real.clamp(0,V-1)[:,:,None]).squeeze(-1)
loss = -target_logp.mean()
loss.backward()
grad_norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None)**0.5
print(f"  Loss: {loss.item():.4f}, grad_norm: {grad_norm:.6f}")
model.zero_grad()
model.eval()
print("  DIFFERENTIABLE PATH: OK")

# Coupling measurement on real checkpoint
print("\n=== Coupling on real checkpoint ===")
torch.manual_seed(42)
B, L = 4, 64
x0_batch = torch.randint(0, V, (B, L), device="cuda")
onehot_batch = F.one_hot(x0_batch, V).float()

t_vals = [0.2, 0.5, 0.8]
coupling_results = []
for t_val in t_vals:
    t_vec = torch.full((B,), t_val, device="cuda")
    dalpha, alpha = model.noise(t_vec)
    move_prob = (1 - alpha).view(B, 1).clamp(0, 1)
    sigma_val = model.get_continuous_from_discrete_noise((1-alpha).float()).reshape(B).float()
    
    move = torch.rand(B, L, device="cuda") < move_prob
    random_tokens = torch.randint(0, V, (B, L), device="cuda")
    xt_tokens = torch.where(move, random_tokens, x0_batch)
    reveal = (xt_tokens == x0_batch).float()
    noise_eps2 = torch.randn(B, L, V, device="cuda")
    xt_cont = onehot_batch + sigma_val[:, None, None] * noise_eps2
    xt = onehot_batch * reveal[:,:,None] + (1 - reveal)[:,:,None] * xt_cont
    
    with torch.no_grad():
        logp = model.forward(xt=xt, discrete_noise=(1-alpha).float(), reveal_mask=reveal, continuous_noise=sigma_val)
    pi = logp.exp()
    
    masked = (reveal == 0)
    if masked.any():
        pi_masked = pi[masked]
        xt_masked = xt_cont[masked]
        x0_masked = x0_batch[masked]
        sigma_m = sigma_val.unsqueeze(1).expand(B, L)[masked]
        
        dt = 0.1
        s_val = t_val - dt
        lam = dt / t_val
        bar_lam = s_val / t_val
        
        sigma_s_val = model.get_continuous_from_discrete_noise(
            torch.tensor([1.0 - (1 - 0.001) * s_val], device="cuda")).float().item()
        
        # Measure: does the posterior-weighted Gaussian mixture differ 
        # from a single Gaussian at the expected mean?
        # This is the coupling: log M_theta(x_s|x_t) vs log N(x_s; E[mu], v)
        N_eval = min(int(masked.sum().item()), 200)
        topk = 16  # Use top-16 tokens for mixture
        
        deltas = []
        for i in range(N_eval):
            sigma_t_i = sigma_m[i].item()
            a_i = (sigma_s_val**2) / (sigma_t_i**2 + 1e-8)
            v_i = max((sigma_s_val**2) * (1 - (sigma_s_val**2)/(sigma_t_i**2+1e-8)), 1e-8)
            
            x_s_i = xt_masked[i]
            probs_i, idx_i = pi_masked[i].topk(topk)
            probs_i = probs_i / probs_i.sum()
            
            # log sum_y pi(y) N(x_s; mu_y, v*I)
            # mu_y = (1-a)*e_y + a*x_t
            log_components = []
            weighted_mu = torch.zeros(V, device="cuda")
            for j in range(topk):
                y = idx_i[j].item()
                e_y = torch.zeros(V, device="cuda"); e_y[y] = 1.0
                mu_y = (1 - a_i) * e_y + a_i * x_s_i
                diff = x_s_i - mu_y
                log_g = -0.5 * (V * math.log(2*math.pi*v_i) + (diff**2).sum().item() / v_i)
                log_components.append(log_g + math.log(max(probs_i[j].item(), 1e-30)))
                weighted_mu += probs_i[j] * mu_y
            
            log_mixture = torch.logsumexp(torch.tensor(log_components), 0).item()
            
            # Factored: single Gaussian at weighted mean
            diff_f = x_s_i - weighted_mu
            log_single = -0.5 * (V * math.log(2*math.pi*v_i) + (diff_f**2).sum().item() / v_i)
            
            deltas.append(log_mixture - log_single)
        
        deltas_t = torch.tensor(deltas)
        coupling_results.append({
            "t": t_val,
            "n_masked": int(masked.sum().item()),
            "n_eval": N_eval,
            "coupling_mean": float(deltas_t.mean()),
            "coupling_std": float(deltas_t.std()),
            "coupling_max": float(deltas_t.max()),
            "coupling_min": float(deltas_t.min()),
        })
        print(f"  t={t_val}: coupling mean={deltas_t.mean():.4f} std={deltas_t.std():.4f} max={deltas_t.max():.4f}")

# Save results
output = {
    "gate1": {"total": result["joint_elbo"], "disc": result["discrete_elbo"],
              "cont": result["continuous_logprob"], "shuffled": shuf_result["joint_elbo"],
              "contrast": delta, "passed": delta > 0},
    "differentiable_path": {"loss": loss.item(), "grad_norm": grad_norm},
    "coupling_on_real_model": coupling_results,
}
out_path = "/home/ec2-user/together/shared/artifacts/gate1_coupling_fable.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {out_path}")
print(json.dumps(output, indent=2))
print("ALL DONE")
