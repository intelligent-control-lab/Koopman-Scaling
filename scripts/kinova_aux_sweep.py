"""Auxiliary-loss study on the real Kinova data (normalized): baseline, +Cov, +Ctrl, +Both with weights
w in {0.01, 0.1, 1} (w_cov = w_ctrl for +Both), n_mult=16, m in {4000, 140000} windows, five seeds.
Env: SHARD, NSHARDS, NGPU, PARALLEL. Each run writes ../log/Kinaux__<arm>_w<w>__m<m>_s<seed>/."""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
SHARD, NSHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
NGPU, PARALLEL = int(os.environ.get("NGPU", 1)), int(os.environ.get("PARALLEL", 1))
PY = os.environ.get("PY", sys.executable)
arms = [("base", None)] + [(a, w) for a in ("cov", "ctrl", "both") for w in (0.01, 0.1, 1.0)]
runs = [(a, w, m, s) for s in (17382, 76849, 20965, 84902, 51194) for m in (4000, 140000) for a, w in arms]
runs = [r for i, r in enumerate(runs) if i % NSHARDS == SHARD]
procs, k = [], 0
for a, w, m, s in runs:
    proj = f"Kinaux__{a}_w{w}__m{m}_s{s}"
    log = os.path.join(HERE, "..", "log", proj, "koopman_results_log.csv")
    if os.path.exists(log) and os.path.getsize(log) > 0:
        continue
    while len([p for p in procs if p.poll() is None]) >= PARALLEL:
        time.sleep(5)
    flags = []
    if a in ("cov", "both"): flags += ["--use_covariance_loss", "--cov_loss_weight", str(w)]
    if a in ("ctrl", "both"): flags += ["--use_control_loss", "--ctrl_loss_weight", str(w)]
    cmd = [PY, "train_model.py", "--project_name", proj, "--env_name", "Kinova", "--sample_size", str(m),
           "--encode_dim", "16", "--layer_depth", "3", "--hidden_dim", "256", "--seed", str(s),
           "--use_residual", "--multiply_encode_by_input_dim"] + flags
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k % NGPU), KOOPMAN_NORMALIZE="1")
    fh = open(os.path.join(HERE, "..", "log", proj + ".train.txt"), "w")
    procs.append(subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)); k += 1
    print(f"[kinaux] train {proj}", flush=True)
for p in procs:
    p.wait()
print("[kinaux] done", flush=True)
