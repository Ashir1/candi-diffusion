"""Incoherence sweep: S across L, n_mask, and all 3 checkpoints.

Measures order sensitivity S = Var_sigma[R_token] at every combination of:
  L in {32, 48, 64, 128}
  n_mask in {4, 8, 12}
  model in {candi, dream, llada}
  G = 24 random orders, 16 sequences per config
Writes incoherence_sweep.json.
"""
import sys, os, json, math, time, argparse
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import torch, torch.nn.functional as F
import numpy as np

OUT = "/home/ec2-user/together/shared/artifacts/incoherence"
GORD, NSEQ, SEED = 24, 16, 909

TEXT = ("The committee reviewed the proposal and found several issues with the "
    "methodology, particularly the treatment of missing data in the survey. "
    "Machine learning systems trained on large text corpora exhibit surprising "
    "generalization to tasks they were never explicitly taught to perform. "
    "The river wound through the valley, past orchards heavy with fruit and "
    "small villages whose names appeared on no map of the region. "
    "In quantum mechanics, the measurement problem concerns how definite "
    "outcomes emerge from superpositions when an observation is made. "
    "Economic historians disagree about the causes of the industrial "
    "revolution and the role that patent law played in accelerating it. ") * 500

def measure_S(model_fn, tok_fn, L, n_mask, mask_id=None):
    """Measure S for one (model, L, n_mask) config."""
    ids = tok_fn(TEXT)
    rng = np.random.default_rng(SEED)
    starts = rng.permutation(max(1, len(ids) - L))[:NSEQ]
    seqs = [ids[s:s+L] for s in starts if len(ids[s:s+L]) == L][:NSEQ]
    results = []
    for si, seq in enumerate(seqs):
        x0 = torch.tensor(seq, device="cuda")
        mask_pos = sorted(rng.choice(L, size=n_mask, replace=False).tolist())
        orders = []
        seen = set()
        orng = np.random.default_rng(SEED + si)
        while len(orders) < GORD:
            o = tuple(orng.permutation(mask_pos).tolist())
            if o not in seen: seen.add(o); orders.append(list(o))
        rewards = []
        for order in orders:
            R = model_fn(x0, mask_pos, order, mask_id)
            rewards.append(R)
        S = float(np.var(rewards))
        results.append({"S": S, "reward_mean": float(np.mean(rewards)),
                        "reward_range": float(np.max(rewards)-np.min(rewards))})
    Ss = [r["S"] for r in results]
    return {"S_mean": float(np.mean(Ss)), "S_median": float(np.median(Ss)),
            "S_max": float(np.max(Ss)), "n_seqs": len(results)}

# ---- CANDI model function ----
def make_candi_fn(model, V):
    import joint_elbo as je
    def fn(x0, mask_pos, order, mask_id_unused):
        onehot = F.one_hot(x0, V).double()
        n_mask = len(mask_pos)
        L = x0.shape[0]
        taus = np.linspace(0.98, 0.02, n_mask + 1)
        sig = model.get_continuous_from_discrete_noise(
            torch.tensor(taus, dtype=torch.float32, device="cuda")).double().cpu().numpy()
        gen = torch.Generator(device="cuda"); gen.manual_seed(SEED + hash(tuple(order)) % 10000)
        z = {q: onehot[q] + sig[0]*torch.randn(V, generator=gen, device="cuda", dtype=torch.float64) for q in mask_pos}
        revealed = set(range(L)) - set(mask_pos)
        still = list(mask_pos); R = 0.0
        with torch.no_grad():
            for j in range(1, n_mask+1):
                reveal = torch.zeros(L, device="cuda"); xt = torch.zeros(L, V, device="cuda")
                for p in range(L):
                    if p in revealed: reveal[p]=1.0; xt[p]=onehot[p].float()
                    else: xt[p]=z[p].float()
                logp = model.forward(xt=xt.unsqueeze(0),
                    discrete_noise=torch.tensor([float(taus[j-1])], device="cuda"),
                    reveal_mask=reveal.unsqueeze(0),
                    continuous_noise=torch.tensor([float(sig[j-1])], device="cuda"))[0].double()
                i_j = order[j-1]
                R += float(logp[i_j, x0[i_j]])
                aj = (sig[j]**2)/(sig[j-1]**2); vj = max(sig[j]**2*(1-sig[j]**2/sig[j-1]**2), 1e-12)
                for q in list(z.keys()):
                    if q == i_j: del z[q]; continue
                    e_y = onehot[q]
                    z[q] = e_y + aj*(z[q]-e_y) + math.sqrt(vj)*torch.randn(V, generator=gen, device="cuda", dtype=torch.float64)
                revealed.add(i_j)
        return R
    return fn

# ---- HF masked-LM model function ----
def make_hf_fn(model, mask_id):
    def fn(x0, mask_pos, order, mask_id_arg):
        mid = mask_id_arg or mask_id
        ids = x0.clone()
        for q in mask_pos: ids[q] = mid
        R = 0.0
        with torch.no_grad():
            for j in range(len(order)):
                out = model(ids.unsqueeze(0))
                logits = out.logits if hasattr(out, "logits") else out[0]
                lp = torch.log_softmax(logits[0].float(), -1)
                i_j = order[j]
                R += float(lp[i_j, x0[i_j]])
                ids[i_j] = x0[i_j]
        return R
    return fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["candi", "dream", "llada"], required=True)
    args = ap.parse_args()
    import joint_elbo as je

    Ls = [32, 48, 64, 128]
    n_masks = [4, 8, 12]

    if args.model == "candi":
        model, tok, cfg, li = je.load_real_model(device="cuda", length=None)
        model.eval()
        V = je._real_vocab(model)
        model_fn = make_candi_fn(model, V)
        tok_fn = lambda t: tok(t, add_special_tokens=False)["input_ids"]
        mask_id = None
    else:
        from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
        repo = {"dream": "Dream-org/Dream-v0-Base-7B", "llada": "GSAI-ML/LLaDA-8B-Base"}[args.model]
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        try:
            model = AutoModel.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        except:
            model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        mask_id = tok.mask_token_id or (126336 if args.model == "llada" else tok.convert_tokens_to_ids("<|mask|>"))
        model_fn = make_hf_fn(model, mask_id)
        tok_fn = lambda t: tok(t, add_special_tokens=False)["input_ids"]

    results = {}
    for L in Ls:
        for n_m in n_masks:
            if n_m >= L: continue
            key = f"L{L}_n{n_m}"
            print(f"{args.model} {key}...", end=" ", flush=True)
            r = measure_S(model_fn, tok_fn, L, n_m, mask_id)
            results[key] = r
            print(f"S_mean={r['S_mean']:.4f}", flush=True)

    json.dump({"model": args.model, "configs": results, "G": GORD, "n_seqs": NSEQ},
              open(f"{OUT}/sweep_{args.model}.json", "w"), indent=1)
    print(f"saved sweep_{args.model}.json")

if __name__ == "__main__":
    main()
