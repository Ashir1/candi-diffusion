#!/bin/bash
cd /home/ec2-user/together/ashir/candi-diffusion
export PYTHONPATH=/home/ec2-user/trajectoryforge/venv/lib64/python3.9/site-packages:.
for seed in 0 1 2; do
  python3 /home/ec2-user/together/shared/artifacts/exp1b_train.py --arm red --seed $seed --steps 2000 >> /home/ec2-user/together/shared/artifacts/exp1b/sweep_g5.log 2>&1
done
echo "G5 SWEEP COMPLETE $(date -u)" >> /home/ec2-user/together/shared/artifacts/exp1b/sweep_g5.log
