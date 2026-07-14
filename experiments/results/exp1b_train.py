"""Experiment 1b: fixed order-policy training.

Fixes vs exp1: (1) G=64 rollouts per sequence per step; (2) confidence
warm-start: phase 0 supervised-trains the policy to imitate the confidence
heuristic (reveal argmax of maxlogp feature) before GRPO; (3) 2000 GRPO steps
default; (4) eval includes the confidence heuristic baseline.
Frozen CANDI; policy-only; checkpoints every 100; reversible.
Usage: python3 exp1b_train.py --arm {red,token} --seed N [--steps N] [--pilot]
"""
import sys, os, json, math, time, argparse
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import torch, torch.nn as nn
import numpy as np
import joint_elbo as je

OUT = "/home/ec2-user/together/shared/artifacts/exp1b"
os.makedirs(OUT, exist_ok=True)

L, NMASK, G, BSEQ = 32, 8, 64, 2
EPS_T = 0.02
CLIP = 0.2
LR = 1e-3
WARMSTART_STEPS = 300
WALLCAP = 3600 * 8

def build_grid(model, dev):
    taus = np.linspace(0.98, EPS_T, NMASK + 1)
    sig = model.get_continuous_from_discrete_noise(
        torch.tensor(taus, dtype=torch.float32, device=dev)).double().cpu().numpy()
    lam = [(taus[j-1]-taus[j])/taus[j-1] for j in range(1, NMASK+1)]
    a = [(sig[j]**2)/(sig[j-1]**2) for j in range(1, NMASK+1)]
    v = [max(sig[j]**2*(1-sig[j]**2/sig[j-1]**2), 1e-12) for j in range(1, NMASK+1)]
    return taus, sig, lam, a, v

class OrderPolicy(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, feats):
        return self.net(feats).squeeze(-1)

def make_data(tok, dev, n_train=64, n_eval=16, seed=1234):
    rng = np.random.default_rng(seed)
    text = ("The committee reviewed the proposal and found several issues with the "
        "methodology, particularly the treatment of missing data in the survey. "
        "Machine learning systems trained on large text corpora exhibit surprising "
        "generalization to tasks they were never explicitly taught to perform. "
        "The river wound through the valley, past orchards heavy with fruit and "
        "small villages whose names appeared on no map of the region. "
        "In quantum mechanics, the measurement problem concerns how definite "
        "outcomes emerge from superpositions when an observation is made. "
        "Economic historians disagree about the causes of the industrial "
        "revolution and the role that patent law played in accelerating it. ") * 500
    ids = tok(text, add_special_tokens=False)["input_ids"]
    need = (n_train + n_eval) * L
    assert len(ids) >= need
    starts = rng.permutation(len(ids) - L)[: n_train + n_eval]
    seqs = [torch.tensor(ids[s:s+L], device=dev) for s in starts]
    return seqs[:n_train], seqs[n_train:]

def feats_from_logp(lp, P, j, dev):
    probs = lp.exp()
    ent = -(probs * lp).sum(-1)
    mx = lp.max(-1).values
    t2 = lp.topk(2, -1).values
    margin = t2[:, 0] - t2[:, 1]
    posf = torch.tensor([p / L for p in P], device=dev, dtype=torch.float64)
    stepf = torch.full((len(P),), j / NMASK, device=dev, dtype=torch.float64)
    return torch.stack([ent, mx, margin, posf, stepf], -1).float()

def rollout_group(model, policy, x0, grid, dev, gen, greedy=False, force_confidence=False):
    """G teacher-forced rollouts. Returns orders, policy logprob (with grad),
    R_token, core, and per-step (feats, chosen_idx) for warm-start SFT."""
    taus, sig, lam, a, v = grid
    V = je._real_vocab(model)
    onehot = torch.nn.functional.one_hot(x0, V).double()
    mask_pos = list(range(4, 4 + NMASK))
    revealed = [set(range(L)) - set(mask_pos) for _ in range(G)]
    z = torch.stack([onehot.clone() for _ in range(G)])
    for q in mask_pos:
        eps0 = torch.randn(G, V, generator=gen, device=dev, dtype=torch.float64)
        z[:, q] = onehot[q].unsqueeze(0) + sig[0] * eps0
    still = [list(mask_pos) for _ in range(G)]
    logp_pol = torch.zeros(G, device=dev)
    Rtok = torch.zeros(G, dtype=torch.float64, device=dev)
    core = torch.zeros(G, dtype=torch.float64, device=dev)
    orders = [[] for _ in range(G)]
    sft_pairs = []
    for j in range(1, NMASK + 1):
        reveal = torch.zeros(G, L, device=dev)
        xt = torch.zeros(G, L, V, device=dev)
        for g in range(G):
            for p in range(L):
                if p in revealed[g]: reveal[g, p] = 1.0; xt[g, p] = onehot[p].float()
                else: xt[g, p] = z[g, p].float()
        with torch.no_grad():
            logp = model.forward(xt=xt, discrete_noise=torch.full((G,), float(taus[j-1]), device=dev),
                                 reveal_mask=reveal,
                                 continuous_noise=torch.full((G,), float(sig[j-1]), device=dev)).double()
        aj, vj = a[j-1], v[j-1]
        for g in range(G):
            P = still[g]
            feats = feats_from_logp(logp[g][P], P, j, dev)
            scores = policy(feats.unsqueeze(0)).squeeze(0)
            logits = torch.log_softmax(scores, 0)
            if force_confidence:
                k = int(feats[:, 1].argmax())  # maxlogp = confidence
            elif greedy:
                k = int(torch.argmax(logits))
            else:
                k = int(torch.multinomial(logits.exp(), 1, generator=gen))
            i_j = P[k]
            logp_pol[g] = logp_pol[g] + logits[k]
            sft_pairs.append((feats.detach(), int(feats[:, 1].argmax())))
            orders[g].append(i_j)
            Rtok[g] += logp[g, i_j, x0[i_j]]
            for q in P:
                if q == i_j: continue
                zq = z[g, q]; e_y = onehot[q]
                eps = torch.randn(V, generator=gen, device=dev, dtype=torch.float64)
                z_new = e_y + aj*(zq - e_y) + math.sqrt(vj)*eps
                w = z_new - aj*zq
                wn2 = float((w*w).sum())
                sc = logp[g, q] - (wn2 - 2*(1-aj)*w + (1-aj)**2)/(2*vj)
                core[g] += torch.logsumexp(sc, 0)
                z[g, q] = z_new
            still[g] = [q for q in P if q != i_j]
            revealed[g].add(i_j)
    return orders, logp_pol, Rtok, core, sft_pairs

def train(arm, seed, steps, pilot=False):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = "cuda"
    model, tok, cfg, li = je.load_real_model(device=dev, length=None)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    grid = build_grid(model, dev)
    train_seqs, eval_seqs = make_data(tok, dev)
    policy = OrderPolicy().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    gen = torch.Generator(device=dev); gen.manual_seed(seed * 7919 + 13)
    tag = f"{arm}_s{seed}"
    t0 = time.time()

    # PHASE 0: confidence warm-start (supervised imitation)
    print(f"[{tag}] warm-start: {WARMSTART_STEPS} SFT steps", flush=True)
    ce = nn.CrossEntropyLoss()
    for w in range(WARMSTART_STEPS):
        si = np.random.randint(len(train_seqs))
        _, _, _, _, pairs = rollout_group(model, policy, train_seqs[si], grid, dev, gen, force_confidence=True)
        loss = torch.tensor(0.0, device=dev)
        nb = 0
        for feats, target in pairs[:64]:
            scores = policy(feats.unsqueeze(0)).squeeze(0)
            loss = loss + ce(scores.unsqueeze(0), torch.tensor([target], device=dev))
            nb += 1
        loss = loss / max(nb, 1)
        opt.zero_grad(); loss.backward(); opt.step()
        if w % 100 == 0:
            print(f"  [warm {w}] imitation loss {float(loss):.4f}", flush=True)
    # reset optimizer for RL phase
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)

    # PHASE 1: GRPO
    logs = []
    if pilot: steps = min(steps, 150)
    for step in range(1, steps + 1):
        if time.time() - t0 > WALLCAP:
            print("WALLCAP"); break
        idx = np.random.randint(0, len(train_seqs), BSEQ)
        loss_terms = []
        rew_means = []
        for si in idx:
            x0 = train_seqs[si]
            orders, lp_new, Rtok, core, _ = rollout_group(model, policy, x0, grid, dev, gen)
            R = (Rtok + core) if arm == "red" else Rtok
            adv = ((R - R.mean()) / R.std().clamp_min(1e-6)).float().detach()
            ratio = torch.exp(lp_new - lp_new.detach())
            surr = torch.minimum(ratio * adv, ratio.clamp(1-CLIP, 1+CLIP) * adv)
            loss_terms.append(-surr.mean())
            rew_means.append(float(R.mean()))
        loss = torch.stack(loss_terms).mean()
        if not torch.isfinite(loss):
            json.dump({"step": step, "nan": True}, open(f"{OUT}/{tag}_NAN.json", "w")); break
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        logs.append({"step": step, "reward_mean": float(np.mean(rew_means)),
                     "loss": float(loss), "grad_norm": float(gn), "elapsed": time.time()-t0})
        if step % 10 == 0 or step == 1:
            print(f"[{tag} {step}] R={logs[-1]['reward_mean']:.2f} grad={float(gn):.3f} t={logs[-1]['elapsed']:.0f}s", flush=True)
        if step % 100 == 0:
            torch.save({"policy": policy.state_dict(), "opt": opt.state_dict(), "step": step},
                       f"{OUT}/ckpt_{tag}_{step}.pt")
    eval_res = evaluate(model, policy, eval_seqs, grid, dev)
    final = {"arm": arm, "seed": seed, "steps_done": len(logs), "warmstart_steps": WARMSTART_STEPS,
             "G": G, "logs": logs, "eval": eval_res, "wallclock": time.time()-t0}
    json.dump(final, open(f"{OUT}/result_{tag}.json", "w"), indent=1)
    torch.save({"policy": policy.state_dict()}, f"{OUT}/final_{tag}.pt")
    print("DONE", tag, json.dumps(eval_res, indent=1), flush=True)

@torch.no_grad()
def forced_rewards(model, x0, grid, dev, order, gen):
    taus, sig, lam, a, v = grid
    V = je._real_vocab(model)
    onehot = torch.nn.functional.one_hot(x0, V).double()
    mask_pos = list(range(4, 4 + NMASK))
    revealed = set(range(L)) - set(mask_pos)
    z = {q: onehot[q] + sig[0]*torch.randn(V, generator=gen, device=dev, dtype=torch.float64) for q in mask_pos}
    still = list(mask_pos)
    Rtok = 0.0; core = 0.0
    for j in range(1, NMASK + 1):
        reveal = torch.zeros(1, L, device=dev); xt = torch.zeros(1, L, V, device=dev)
        for p in range(L):
            if p in revealed: reveal[0,p]=1.0; xt[0,p]=onehot[p].float()
            else: xt[0,p]=z[p].float()
        logp = model.forward(xt=xt, discrete_noise=torch.tensor([float(taus[j-1])], device=dev),
                             reveal_mask=reveal,
                             continuous_noise=torch.tensor([float(sig[j-1])], device=dev))[0].double()
        aj, vj = a[j-1], v[j-1]
        i_j = order[j-1]
        Rtok += float(logp[i_j, x0[i_j]])
        for q in still:
            if q == i_j: continue
            zq = z[q]; e_y = onehot[q]
            eps = torch.randn(V, generator=gen, device=dev, dtype=torch.float64)
            z_new = e_y + aj*(zq - e_y) + math.sqrt(vj)*eps
            w = z_new - aj*zq
            wn2 = float((w*w).sum())
            sc = logp[q] - (wn2 - 2*(1-aj)*w + (1-aj)**2)/(2*vj)
            core += float(torch.logsumexp(sc, 0))
            z[q] = z_new
        still = [q for q in still if q != i_j]
        revealed.add(i_j)
    return Rtok, Rtok + core

@torch.no_grad()
def confidence_order(model, x0, grid, dev, gen):
    """Greedy confidence heuristic order for baseline."""
    taus, sig, lam, a, v = grid
    V = je._real_vocab(model)
    onehot = torch.nn.functional.one_hot(x0, V).double()
    mask_pos = list(range(4, 4 + NMASK))
    revealed = set(range(L)) - set(mask_pos)
    z = {q: onehot[q] + sig[0]*torch.randn(V, generator=gen, device=dev, dtype=torch.float64) for q in mask_pos}
    still = list(mask_pos); order = []
    for j in range(1, NMASK + 1):
        reveal = torch.zeros(1, L, device=dev); xt = torch.zeros(1, L, V, device=dev)
        for p in range(L):
            if p in revealed: reveal[0,p]=1.0; xt[0,p]=onehot[p].float()
            else: xt[0,p]=z[p].float()
        logp = model.forward(xt=xt, discrete_noise=torch.tensor([float(taus[j-1])], device=dev),
                             reveal_mask=reveal,
                             continuous_noise=torch.tensor([float(sig[j-1])], device=dev))[0].double()
        aj, vj = a[j-1], v[j-1]
        mx = {q: float(logp[q].max()) for q in still}
        i_j = max(mx, key=mx.get)
        order.append(i_j)
        for q in still:
            if q == i_j: continue
            zq = z[q]; e_y = onehot[q]
            eps = torch.randn(V, generator=gen, device=dev, dtype=torch.float64)
            z[q] = e_y + aj*(zq - e_y) + math.sqrt(vj)*eps
        still = [q for q in still if q != i_j]
        revealed.add(i_j)
    return order

@torch.no_grad()
def evaluate(model, policy, eval_seqs, grid, dev):
    gen = torch.Generator(device=dev); gen.manual_seed(999)
    rng = np.random.default_rng(999)
    mask_pos = list(range(4, 4 + NMASK))
    out = {k: [] for k in ["pol_red","pol_tok","conf_red","conf_tok","l2r_red","l2r_tok","rnd_red","rnd_tok"]}
    for x0 in eval_seqs:
        orders, _, Rtok, core, _ = rollout_group(model, policy, x0, grid, dev, gen, greedy=True)
        out["pol_red"].append(float((Rtok+core)[0])); out["pol_tok"].append(float(Rtok[0]))
        co = confidence_order(model, x0, grid, dev, gen)
        t_, r_ = forced_rewards(model, x0, grid, dev, co, gen)
        out["conf_tok"].append(t_); out["conf_red"].append(r_)
        t_, r_ = forced_rewards(model, x0, grid, dev, list(mask_pos), gen)
        out["l2r_tok"].append(t_); out["l2r_red"].append(r_)
        ts, rs = [], []
        for _ in range(4):
            o = list(rng.permutation(mask_pos))
            t2, r2 = forced_rewards(model, x0, grid, dev, o, gen)
            ts.append(t2); rs.append(r2)
        out["rnd_tok"].append(float(np.mean(ts))); out["rnd_red"].append(float(np.mean(rs)))
    return {k: float(np.mean(vv)) for k, vv in out.items()}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["red", "token"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    train(args.arm, args.seed, args.steps, pilot=args.pilot)
