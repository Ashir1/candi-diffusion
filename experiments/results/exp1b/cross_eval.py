"""Cross-eval: all 6 final policies under ONE eval configuration (this g5).
Kills the arm-x-machine confound. Also evals the warm-start-only policy
(confidence-imitation, 0 GRPO steps) as a control."""
import sys, json, glob
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import torch
import exp1b_train as T
import joint_elbo as je

dev = "cuda"
model, tok, cfg, li = je.load_real_model(device=dev, length=None)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
grid = T.build_grid(model, dev)
_, eval_seqs = T.make_data(tok, dev)

out = {}
for f in sorted(glob.glob("/home/ec2-user/together/shared/artifacts/exp1b/final_*.pt")):
    tag = f.split("final_")[1].replace(".pt", "")
    pol = T.OrderPolicy().to(dev)
    pol.load_state_dict(torch.load(f, map_location=dev)["policy"])
    pol.eval()
    res = T.evaluate(model, pol, eval_seqs, grid, dev)
    out[tag] = res
    print(tag, json.dumps({k: round(v, 3) for k, v in res.items()}))

json.dump(out, open("/home/ec2-user/together/shared/artifacts/exp1b/cross_eval.json", "w"), indent=1)
print("saved cross_eval.json")
