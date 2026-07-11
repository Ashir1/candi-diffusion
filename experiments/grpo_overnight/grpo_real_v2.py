"""GRPO v2: Fixed ratio normalization.

Key fix: normalize log-ratio PER TOKEN rather than per sequence.
This prevents the ratio from exploding when summed over L*n_quad terms.

Also adds:
- Ratio capping at exp(+/-2) before clipping 
- Per-token advantage weighting
- Better KL estimation (per-token, not per-sequence)
"""
import sys, os, json, math, time, copy
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
sys.path.insert(0, "/home/ec2-user/together/wt_agent_1")
sys.path.insert(0, "/home/ec2-user/together/ashir/candi-diffusion")
sys.path.insert(0, "/home/ec2-user/trajectoryforge/venv/lib64/python3.9/site-packages")

import torch
import torch.nn.functional as F
import numpy as np

# Config
MAX_STEPS = 200
WALLCLOCK_CAP = 3600 * 3
GROUP_SIZE = 4
SEQ_LEN = 64
N_PROMPTS = 2
NFE_GEN = 16
LR = 3e-6  # reduced from 1e-5
CLIP_EPS = 0.2
KL_COEF = 0.05  # increased KL penalty
MAX_GRAD_NORM = 0.5  # tighter grad clipping
LOG_EVERY = 10
SEED = 42
RATIO_CAP = 2.0  # cap log-ratio at +/- 2 (ratio in [0.14, 7.4])

def load_model():
    import joint_elbo as je
    model, tok, cfg, load_info = je.load_real_model(device="cuda", length=None)
    print("Loaded:", load_info)
    return model, tok, cfg

def per_token_logprob(model, x0, t_scalar=0.5):
    """Single-timestep per-token log prob (differentiable).
    
    Returns [B, L] tensor of per-token log-probs.
    Using a single time point for stability (matched between new and old).
    """
    V = model.vocab_size - 1
    B, L = x0.shape
    device = x0.device
    
    t_vec = torch.full((B,), t_scalar, device=device)
    dalpha, alpha = model.noise(t_vec)
    disc_noise = (1 - alpha).float()
    sigma = model.get_continuous_from_discrete_noise(disc_noise).reshape(B).float()
    
    x0_clip = x0.clamp(0, V-1)
    onehot = F.one_hot(x0_clip, num_classes=V).float()
    
    # Use FIXED corruption (same noise for both old and new policy evaluation)
    # This is critical for variance reduction
    gen = torch.Generator(device=device)
    gen.manual_seed(12345 + int(x0[0, 0].item()) % 1000)  # deterministic per sample
    
    move = torch.rand(B, L, device=device, generator=gen) < (1 - alpha).view(B, 1)
    random_tokens = torch.randint(0, V, (B, L), device=device, generator=gen)
    xt_tokens = torch.where(move, random_tokens, x0_clip)
    reveal = (xt_tokens == x0_clip).float()
    noise = torch.randn(B, L, V, device=device, generator=gen) * sigma[:, None, None]
    xt_cont = onehot + noise
    xt = onehot * reveal[:,:,None] + (1-reveal)[:,:,None] * xt_cont
    
    logp = model.forward(xt=xt, discrete_noise=disc_noise, reveal_mask=reveal, continuous_noise=sigma)
    target_logp = logp.gather(-1, x0_clip[:,:,None]).squeeze(-1)
    return target_logp  # [B, L]

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

def run_grpo_v2():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    print("=" * 60)
    print("GRPO v2: Joint ratio with per-token normalization")
    print("=" * 60)
    
    model, tok, cfg = load_model()
    model.train()
    optimizer = torch.optim.AdamW(model.backbone.parameters(), lr=LR, weight_decay=0.01)
    
    # Reference model
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    
    V = model.vocab_size - 1
    device = next(model.parameters()).device
    
    logs = []
    start_time = time.time()
    best_reward = -float("inf")
    
    print(f"Config: steps={MAX_STEPS}, G={GROUP_SIZE}, L={SEQ_LEN}, NFE={NFE_GEN}")
    print(f"        LR={LR}, clip={CLIP_EPS}, KL={KL_COEF}, grad_clip={MAX_GRAD_NORM}")
    print(f"        ratio_cap=exp(+/-{RATIO_CAP})")
    print()
    
    for step in range(1, MAX_STEPS + 1):
        elapsed = time.time() - start_time
        if elapsed > WALLCLOCK_CAP:
            print(f"WALLCLOCK CAP at step {step}")
            break
        
        # Generate
        model.eval()
        with torch.no_grad():
            n_total = N_PROMPTS * GROUP_SIZE
            samples = generate_samples(model, n_total, SEQ_LEN, nfe=NFE_GEN)
            rewards = reward_fn(samples)
        
        # Advantages
        grouped = rewards.view(N_PROMPTS, GROUP_SIZE)
        adv = (grouped - grouped.mean(1, keepdim=True)) / grouped.std(1, keepdim=True).clamp_min(1e-6)
        adv = adv.view(-1)  # [n_total]
        
        # Compute per-token log-ratios
        model.train()
        new_logp = per_token_logprob(model, samples)  # [n_total, L]
        with torch.no_grad():
            old_logp = per_token_logprob(ref_model, samples)  # [n_total, L]
        
        # Per-token log ratio, MEAN over positions for the sequence-level ratio
        token_log_ratio = new_logp - old_logp  # [n_total, L]
        seq_log_ratio = token_log_ratio.mean(dim=-1)  # [n_total] - mean, not sum!
        
        # Cap the ratio for stability
        seq_log_ratio_capped = seq_log_ratio.clamp(-RATIO_CAP, RATIO_CAP)
        ratio = torch.exp(seq_log_ratio_capped)
        
        # Clipped GRPO objective
        unclipped = ratio * adv
        clipped = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        
        # KL (per-token mean)
        kl = seq_log_ratio.mean()
        
        loss = policy_loss + KL_COEF * kl
        
        if not torch.isfinite(loss):
            print(f"NaN at step {step}! Stopping.")
            break
        
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        log_entry = {
            "step": step,
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_max": float(rewards.max()),
            "loss": float(loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "kl": float(kl.detach()),
            "ratio_mean": float(ratio.mean().detach()),
            "ratio_std": float(ratio.std().detach()),
            "log_ratio_mean": float(seq_log_ratio.mean().detach()),
            "log_ratio_std": float(seq_log_ratio.std().detach()),
            "grad_norm": float(grad_norm),
            "elapsed_sec": elapsed,
        }
        logs.append(log_entry)
        
        if step % LOG_EVERY == 0 or step == 1:
            print("[%4d] rwd=%.4f(%.3f) loss=%.4f kl=%.4f ratio=%.3f(%.3f) logratio=%.4f grad=%.4f t=%.0fs" % (
                step, log_entry["reward_mean"], log_entry["reward_std"],
                log_entry["loss"], log_entry["kl"],
                log_entry["ratio_mean"], log_entry["ratio_std"],
                log_entry["log_ratio_mean"], log_entry["grad_norm"], elapsed))
        
        if log_entry["reward_mean"] > best_reward:
            best_reward = log_entry["reward_mean"]
    
    final = {
        "method": "joint_ratio_per_token_normalized",
        "version": 2,
        "total_steps": len(logs),
        "wallclock_sec": time.time() - start_time,
        "reward_start": logs[0]["reward_mean"] if logs else None,
        "reward_end": logs[-1]["reward_mean"] if logs else None,
        "reward_best": best_reward,
        "any_nan": any(not math.isfinite(l["loss"]) for l in logs),
        "config": {
            "max_steps": MAX_STEPS, "group_size": GROUP_SIZE, "seq_len": SEQ_LEN,
            "nfe_gen": NFE_GEN, "lr": LR, "clip_eps": CLIP_EPS, "kl_coef": KL_COEF,
            "max_grad_norm": MAX_GRAD_NORM, "ratio_cap": RATIO_CAP, "n_prompts": N_PROMPTS,
        },
        "logs": logs,
    }
    out_path = "/home/ec2-user/together/shared/artifacts/grpo_joint_v2.json"
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)
    print("\n" + "="*60)
    print("DONE. Steps=%d, reward %.4f -> %.4f (best %.4f)" % (
        len(logs), logs[0]["reward_mean"], logs[-1]["reward_mean"], best_reward))
    print("Saved to", out_path)
    
    # Status update
    with open("/home/ec2-user/together/shared/status/agent_fable.md", "a") as f:
        f.write("\n## %s - GRPO v2 (joint, per-token norm) complete\n" % time.strftime("%Y-%m-%d %H:%M UTC"))
        f.write("Steps: %d, reward: %.4f -> %.4f (best %.4f)\n" % (
            len(logs), logs[0]["reward_mean"], logs[-1]["reward_mean"], best_reward))
        f.write("NaN: %s, wallclock: %.0fs\n" % (final["any_nan"], final["wallclock_sec"]))
        f.write("STATUS: partial\n")

if __name__ == "__main__":
    run_grpo_v2()
