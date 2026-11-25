"""Kinova scaling rerun with the corrected data loader (stride-4 windows, chronological split with gaps).

Grid of the paper's scaling figures: m in {1, 4, 16, 64, 140}k x encode_dim in {1, 2, 4, 8, 16} x 5 seeds = 125 runs.
LOSS=base (no auxiliary losses; the configuration of the paper's scaling figures) or LOSS=both (+Aux).
Env: LOSS, SHARD, NSHARDS, NGPU, PARALLEL. Each run writes ../log/Kinova_v3<base|both>__m<m>_nm<n>_s<seed>/.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHARD, NSHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
NGPU, PARALLEL = int(os.environ.get("NGPU", 1)), int(os.environ.get("PARALLEL", 1))
PY = os.environ.get("PY", sys.executable)
LOSS = os.environ.get("LOSS", "base")
LOSS_FLAGS = [] if LOSS == "base" else ["--use_control_loss", "--use_covariance_loss"]
runs = [(m, nm, s) for s in (17382, 76849, 20965, 84902, 51194) for nm in (1, 2, 4, 8, 16)
        for m in (1000, 4000, 16000, 64000, 140000)]
runs = [r for i, r in enumerate(runs) if i % NSHARDS == SHARD]
procs, k = [], 0
for m, nm, s in runs:
    proj = f"Kinova_v3{LOSS}__m{m}_nm{nm}_s{s}"
    log = os.path.join(HERE, "..", "log", proj, "koopman_results_log.csv")
    if os.path.exists(log) and os.path.getsize(log) > 0:
        continue
    while len([p for p in procs if p.poll() is None]) >= PARALLEL:
        time.sleep(5)
    cmd = [PY, "train_model.py", "--project_name", proj, "--env_name", "Kinova", "--sample_size", str(m),
           "--encode_dim", str(nm), "--layer_depth", "3", "--hidden_dim", "256", "--seed", str(s),
           "--use_residual", "--multiply_encode_by_input_dim"] + LOSS_FLAGS
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % NGPU))
    fh = open(os.path.join(HERE, "..", "log", proj + ".train.txt"), "w")
    procs.append(subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)); k += 1
    print(f"[kinova] train {proj} (gpu {k % NGPU})", flush=True)
for p in procs:
    p.wait()
print("[kinova] done", flush=True)
