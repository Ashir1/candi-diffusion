"""Exp1e: L=64 retry with G=64 (fixing exp1d's insufficient group size).

exp1d at G=32 failed to beat random at n_mask=12. Doubling G halves the
advantage noise; also using longer training (1500 steps) and slightly larger
policy (d=128->256).
"""
import sys
sys.path.insert(0, "/home/ec2-user/together/shared/artifacts")
import exp1b_train as T
import torch.nn as nn

T.L = 64
T.NMASK = 12
T.G = 64
T.BSEQ = 1  # one sequence per step at G=64 to fit memory
T.OUT = "/home/ec2-user/together/shared/artifacts/exp1e"
import os; os.makedirs(T.OUT, exist_ok=True)

# bigger policy head
class BiggerPolicy(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, feats):
        return self.net(feats).squeeze(-1)
T.OrderPolicy = BiggerPolicy

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["red", "token"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    T.train(args.arm, args.seed, args.steps)
