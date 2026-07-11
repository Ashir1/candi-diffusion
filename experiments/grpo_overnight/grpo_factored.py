"""Factored-ratio GRPO baseline for ablation comparison.

Same as grpo_real.py but uses a FACTORED ratio:
- Discrete ratio: pi_new(x0) / pi_old(x0) at each token independently
- Ignores the coupling between discrete and continuous channels

This should show:
1. Biased updates on high-advantage samples
2. Potential instability compared to the joint ratio
"""
import sys, os, json, math, time, copy
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
sys.path.insert(0, "/home/ec2-user/together/wt_agent_1")
sys.path.insert(0, "/home/ec2-user/together/ashir/candi-diffusion")
os.environ["PYTHONPATH"] = "/home/ec2-user/trajectoryforge/venv/lib64/python3.9/site-packages:" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, "/home/ec2-user/trajectoryforge/venv/lib64/python3.9/site-packages")

import torch
import torch.nn.functional as F
import numpy as np

# Same config as joint
MAX_STEPS = 200
WALLCLOCK_CAP = 3600 * 4
GROUP_SIZE = 4
SEQ_LEN = 64
N_PROMPTS = 2
NFE_GEN = 16
LR = 1e-5
CLIP_EPS = 0.2
KL_COEF = 0.01
MAX_GRAD_NORM = 1.0
LOG_EVERY = 5
SAVE_EVERY = 50
SEED = 42

def load_model():
    import joint_elbo as je
    model, tok, cfg, load_info = je.load_real_model(device="cuda", length=None)
    return model, tok, cfg

def factored_discrete_logprob(model, x0):
    """Factored discrete log-prob: just the token prediction loss at a single time point.
    
    This treats discrete and continuous channels independently.
    Uses a single quadrature node at t=0.5 for speed.
    """
    V = model.vocab_size - 1
    B, L = x0.shape
    device = x0.device
    
    t_vec = torch.full((B,), 0.5, device=device)
    dalpha, alpha = model.noise(t_vec)
    disc_noise = (1 - alpha).float()
    sigma = model.get_continuous_from_discrete_noise(disc_noise).reshape(B).float()
    
    x0_clip = x0.clamp(0, V-1)
    onehot = F.one_hot(x0_clip, num_classes=V).float()
    
    # Corrupt
    move = torch.rand(B, L, device=device) < (1 - alpha).view(B, 1)
    random_tokens = torch.randint(0, V, (B, L), device=device)
    xt_tokens = torch.where(move, random_tokens, x0_clip)
    reveal = (xt_tokens == x0_clip).float()
    noise = torch.randn(B, L, V, device=device) * sigma[:, None, None]
    xt_cont = onehot + noise
    xt = onehot * reveal[:,:,None] + (1-reveal)[:,:,None] * xt_cont
    
    # Forward
    logp = model.forward(xt=xt, discrete_noise=disc_noise, reveal_mask=reveal, continuous_noise=sigma)
    
    # FACTORED: just the per-token log prob, ignore continuous coupling
    target_logp = logp.gather(-1, x0_clip[:,:,None]).squeeze(-1)
    return target_logp.sum()  # sum over batch and positions

@torch.no_grad()
def generate_samples(model, n_samples, seq_len, nfe=16):
    V = model.vocab_size - 1
    device = next(model.parameters()).device
    eps = 1e-3
    x = model.prior_sample(n_samples, seq_len)
    clean_mask = torch.zeros(n_samples, seq_len, device=device)
    timesteps = torch.linspace(0.999, eps, nfe + 1, device=device)
    continuous_noise = model.get_continuous_from_discrete_noise(timesteps).float()
    dt = (1 - eps) / nfe
    for i in range(nfe):
        t = timesteps[i]
        sigma_s = continuous_noise[i]
        sigma_t = continuous_noise[i+1]
        s = timesteps[i+1]
        x_cont, p_x0 = model._continuous_step(x, t, sigma_s=sigma_s, sigma_t=sigma_t,
                                               clean_mask=clean_mask, time_s=s)
        x, clean_mask = model._discrete_step(x_cont, p_x0, t, dt, prev_clean_mask=clean_mask)
    return x.argmax(dim=-1)

def reward_fn(tokens):
    threshold = 1000
    common = (tokens < threshold).float()
    return common.mean(-1)

def run_factored_grpo():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("=" * 60)
    print("FACTORED-RATIO GRPO (ablation baseline)")
    print("=" * 60)
    
    model, tok, cfg = load_model()
    model.train()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=LR, weight_decay=0.01)
    
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    
    V = model.vocab_size - 1
    device = next(model.parameters()).device
    
    logs = []
    start_time = time.time()
    best_reward = -float("inf")
    
    print(f"\nConfig: steps={MAX_STEPS}, G={GROUP_SIZE}, L={SEQ_LEN}, NFE={NFE_GEN}")
    print(f"        LR={LR}, clip={CLIP_EPS}, KL={KL_COEF}")
    print(f"        RATIO TYPE: FACTORED (discrete only, no coupling)")
    print()
    
    for step in range(1, MAX_STEPS + 1):
        elapsed = time.time() - start_time
        if elapsed > WALLCLOCK_CAP:
            print(f"WALLCLOCK CAP at step {step}")
            break
        
        model.eval()
        with torch.no_grad():
            n_total = N_PROMPTS * GROUP_SIZE
            samples = generate_samples(model, n_total, SEQ_LEN, nfe=NFE_GEN)
            rewards = reward_fn(samples)
        
        grouped = rewards.view(N_PROMPTS, GROUP_SIZE)
        adv = (grouped - grouped.mean(1, keepdim=True)) / grouped.std(1, keepdim=True).clamp_min(1e-6)
        adv = adv.view(-1)
        
        model.train()
        
        # Per-sample factored log-probs
        per_sample_new = []
        per_sample_old = []
        for i in range(n_total):
            s_i = samples[i:i+1]
            new_lp = factored_discrete_logprob(model, s_i)
            per_sample_new.append(new_lp)
            with torch.no_grad():
                old_lp = factored_discrete_logprob(ref_model, s_i)
            per_sample_old.append(old_lp)
        
        new_logps = torch.stack(per_sample_new)
        old_logps = torch.stack(per_sample_old)
        
        log_ratio = new_logps - old_logps
        ratio = torch.exp(log_ratio.clamp(-5, 5))
        
        unclipped = ratio * adv
        clipped = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        kl = log_ratio.mean()
        loss = policy_loss + KL_COEF * kl
        
        if not torch.isfinite(loss):
            print(f"NaN at step {step}!")
            break
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        log_entry = {
            "step": step, "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()), "loss": float(loss.detach()),
            "kl": float(kl.detach()), "ratio_mean": float(ratio.mean().detach()),
            "ratio_std": float(ratio.std().detach()), "grad_norm": float(grad_norm),
            "elapsed_sec": elapsed,
        }
        logs.append(log_entry)
        
        if step % LOG_EVERY == 0 or step == 1:
            print(f"[{step:4d}] reward={log_entry['reward_mean']:.4f} "
                  f"loss={log_entry['loss']:.4f} kl={log_entry['kl']:.4f} "
                  f"ratio={log_entry['ratio_mean']:.4f} grad={log_entry['grad_norm']:.4f}")
        
        if log_entry["reward_mean"] > best_reward:
            best_reward = log_entry["reward_mean"]
    
    final = {
        "method": "factored_ratio",
        "total_steps": len(logs),
        "wallclock_sec": time.time() - start_time,
        "reward_start": logs[0]["reward_mean"] if logs else None,
        "reward_end": logs[-1]["reward_mean"] if logs else None,
        "reward_best": best_reward,
        "any_nan": any(not math.isfinite(l["loss"]) for l in logs),
        "config": {"max_steps": MAX_STEPS, "group_size": GROUP_SIZE, "seq_len": SEQ_LEN,
                   "nfe_gen": NFE_GEN, "lr": LR, "clip_eps": CLIP_EPS, "kl_coef": KL_COEF},
        "logs": logs,
    }
    out_path = "/home/ec2-user/together/shared/artifacts/grpo_factored_run.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nDONE. Steps={len(logs)}, reward: {logs[0]['reward_mean']:.4f} -> {logs[-1]['reward_mean']:.4f}")
    return final

if __name__ == "__main__":
    run_factored_grpo()
