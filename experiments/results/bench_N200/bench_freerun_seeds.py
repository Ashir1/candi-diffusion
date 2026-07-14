"""Free-running gen-PPL benchmark: all 6 policy seeds (3 red, 3 token) + baselines.
N=200 samples/cell, NFE=32, GPT-2-large scorer. Confidence decoder order."""
import sys, os, json, math, time
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import torch, torch.nn.functional as F
import numpy as np
import joint_elbo as je
import exp1b_train as T

OUT = "/home/ec2-user/together/shared/artifacts/bench"
os.makedirs(OUT, exist_ok=True)
L = 64
NFE = 32
NSAMP = 200
SEED_BASE = 7

dev = "cuda"

def order_scores(rule, policy, logp, still, j, nsteps, dev):
    lp = logp[still]
    if rule == "random":
        return torch.rand(len(still), device=dev)
    if rule == "l2r":
        return -torch.tensor(still, dtype=torch.float32, device=dev)
    if rule == "confidence":
        return lp.max(-1).values.float()
    probs = lp.exp()
    ent = -(probs * lp).sum(-1)
    mx = lp.max(-1).values
    t2 = lp.topk(2, -1).values
    margin = t2[:, 0] - t2[:, 1]
    posf = torch.tensor([p / L for p in still], device=dev, dtype=torch.float64)
    stepf = torch.full((len(still),), j / nsteps, device=dev, dtype=torch.float64)
    feats = torch.stack([ent, mx, margin, posf, stepf], -1).float()
    return policy(feats.unsqueeze(0)).squeeze(0)

@torch.no_grad()
def generate(model, rule, policy, nfe, n, dev, gen):
    V = je._real_vocab(model)
    taus = np.linspace(0.98, 0.02, nfe + 1)
    sig = model.get_continuous_from_discrete_noise(
        torch.tensor(taus, dtype=torch.float32, device=dev)).double().cpu().numpy()
    z = torch.randn(n, L, V, generator=gen, device=dev, dtype=torch.float64) * sig[0]
    revealed = torch.zeros(n, L, dtype=torch.bool, device=dev)
    tokens = torch.zeros(n, L, dtype=torch.long, device=dev)
    for j in range(1, nfe + 1):
        reveal_mask = revealed.float()
        xt = torch.where(revealed.unsqueeze(-1),
                         F.one_hot(tokens, V).double(), z).float()
        logp = model.forward(xt=xt,
                             discrete_noise=torch.full((n,), float(taus[j-1]), device=dev),
                             reveal_mask=reveal_mask,
                             continuous_noise=torch.full((n,), float(sig[j-1]), device=dev)).double()
        remaining = L - int(revealed[0].sum())
        k = math.ceil(remaining / (nfe - j + 1))
        aj = (sig[j]**2) / (sig[j-1]**2)
        vj = max(sig[j]**2 * (1 - sig[j]**2 / sig[j-1]**2), 1e-12)
        for b in range(n):
            still = (~revealed[b]).nonzero().squeeze(-1).tolist()
            if not still: continue
            sc = order_scores(rule, policy, logp[b], still, j, nfe, dev)
            topk = torch.topk(sc, min(k, len(still))).indices.tolist()
            chosen = [still[i] for i in topk]
            for q in chosen:
                y = int(torch.multinomial(logp[b, q].exp(), 1, generator=gen))
                tokens[b, q] = y
                revealed[b, q] = True
            rest = [q for q in still if q not in chosen]
            if rest and j < nfe:
                idx = torch.tensor(rest, device=dev)
                ys = torch.multinomial(logp[b, idx].exp(), 1, generator=gen).squeeze(-1)
                e_y = F.one_hot(ys, V).double()
                mu = e_y + aj * (z[b, idx] - e_y)
                z[b, idx] = mu + math.sqrt(vj) * torch.randn(len(rest), V, generator=gen, device=dev, dtype=torch.float64)
        if j == nfe:
            for b in range(n):
                still = (~revealed[b]).nonzero().squeeze(-1).tolist()
                for q in still:
                    tokens[b, q] = int(torch.multinomial(logp[b, q].exp(), 1, generator=gen))
                    revealed[b, q] = True
    return tokens

def main():
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    model, tok, cfg, li = je.load_real_model(device=dev, length=None)
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)

    sc_tok = GPT2TokenizerFast.from_pretrained("gpt2-large")
    scorer = GPT2LMHeadModel.from_pretrained("gpt2-large").to(dev).eval()

    def score_batch(token_batch):
        texts = tok.batch_decode(token_batch.cpu())
        nlls = []
        for txt in texts:
            ids = sc_tok(txt, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
            if ids.shape[1] < 2: continue
            with torch.no_grad(): nlls.append(float(scorer(ids, labels=ids).loss))
        return nlls

    results = {}
    
    # Load all 6 policy seeds
    for arm in ["red", "token"]:
        for s in range(3):
            path = f"/home/ec2-user/together/shared/artifacts/exp1b/final_{arm}_s{s}.pt"
            if not os.path.exists(path):
                print(f"MISSING {path}", flush=True); continue
            pol = T.OrderPolicy().to(dev)
            pol.load_state_dict(torch.load(path, map_location=dev)["policy"])
            pol.eval()
            
            all_nlls = []
            nbatch = (NSAMP + 31) // 32
            for b in range(nbatch):
                gen = torch.Generator(device=dev); gen.manual_seed(SEED_BASE + b)
                bs = min(32, NSAMP - b*32)
                toks = generate(model, "policy", pol, NFE, bs, dev, gen)
                all_nlls.extend(score_batch(toks))
            
            ppl = float(np.exp(np.mean(all_nlls)))
            std = float(np.std([np.exp(x) for x in all_nlls]))
            key = f"policy_{arm}_s{s}@nfe{NFE}_N{NSAMP}"
            results[key] = {"gen_ppl": ppl, "std": std, "n": len(all_nlls)}
            print(f"{key}: ppl={ppl:.2f} std={std:.2f} n={len(all_nlls)}", flush=True)
            del pol; torch.cuda.empty_cache()
    
    # Baselines: confidence, random, l2r
    for rule in ["confidence", "random", "l2r"]:
        all_nlls = []
        nbatch = (NSAMP + 31) // 32
        for b in range(nbatch):
            gen = torch.Generator(device=dev); gen.manual_seed(SEED_BASE + b)
            bs = min(32, NSAMP - b*32)
            toks = generate(model, rule, None, NFE, bs, dev, gen)
            all_nlls.extend(score_batch(toks))
        ppl = float(np.exp(np.mean(all_nlls)))
        std = float(np.std([np.exp(x) for x in all_nlls]))
        key = f"{rule}@nfe{NFE}_N{NSAMP}"
        results[key] = {"gen_ppl": ppl, "std": std, "n": len(all_nlls)}
        print(f"{key}: ppl={ppl:.2f} std={std:.2f} n={len(all_nlls)}", flush=True)
    
    json.dump(results, open(f"{OUT}/freerun_seeds_N200.json", "w"), indent=1)
    print("FREERUN_SEEDS COMPLETE", flush=True)

if __name__ == "__main__":
    main()
