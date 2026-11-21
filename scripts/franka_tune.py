"""Validation-based tuning of the auxiliary losses on Franka (original Koopman-Scaling training and MPC code).

Protocol (same structure as the G1/Go2 study; nothing is selected on the reported metric):
  select: m = 140k, n_mult in {1, 4, 16}, seeds {17382, 76849}.
          baseline: train_steps in {60k, 120k};  +Both: train_steps x w_ctrl in {0.01, 0.1, 1} x w_cov in {0.1, 1}.
          Settings are chosen by the STAR-trajectory tracking error (lower is better).
  final:  selected settings for Baseline, +Cov, +Ctrl, +Both on m in {1, 4, 16, 64, 140}k x n_mult in {1, 2, 4, 8, 16}
          x 5 seeds; the reported metric is the FIGURE-8 tracking error (the paper's metric).
Every arm is evaluated with the same MPC settings (franka_mpc.FrankaMPCConfig with warmup 50, as for the paper's
Koopman Franka results).

Env: STAGE=select|final, SHARD, NSHARDS, NGPU, PARALLEL (trainings at once), EVAL_PAR (MPC evaluations at once).
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "log")
STAGE = os.environ.get("STAGE", "select")
SHARD, NSHARDS = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARDS", 1))
NGPU, PARALLEL = int(os.environ.get("NGPU", 1)), int(os.environ.get("PARALLEL", 1))
EVAL_PAR = int(os.environ.get("EVAL_PAR", 8))
PY = os.environ.get("PY", sys.executable)
SEEDS5 = [17382, 76849, 20965, 84902, 51194]


def runs():
    out = []
    if STAGE == "select":
        settings = [("base", s, 1.0, 1.0) for s in (60000, 120000)]
        settings += [("both", s, c, u) for s in (60000, 120000) for u in (0.01, 0.1, 1.0) for c in (0.1, 1.0)]
        for arm, steps, wcov, wctrl in settings:
            for nm in (1, 4, 16):
                for seed in SEEDS5[:2]:
                    out.append(dict(arm=arm, steps=steps, wcov=wcov, wctrl=wctrl, m=140000, nm=nm, seed=seed))
    else:
        sel = json.load(open(os.path.join(LOG, "franka_tuning", "selected.json")))
        b, a = sel["baseline"], sel["both"]
        arms = [("base", b["train_steps"], 1.0, 1.0), ("cov", a["train_steps"], a["cov_loss_weight"], 1.0),
                ("ctrl", a["train_steps"], 1.0, a["ctrl_loss_weight"]),
                ("both", a["train_steps"], a["cov_loss_weight"], a["ctrl_loss_weight"])]
        for arm, steps, wcov, wctrl in arms:
            for m in (1000, 4000, 16000, 64000, 140000):
                for nm in (1, 2, 4, 8, 16):
                    for seed in SEEDS5:
                        out.append(dict(arm=arm, steps=steps, wcov=wcov, wctrl=wctrl, m=m, nm=nm, seed=seed))
    return [r for i, r in enumerate(out) if i % NSHARDS == SHARD]


def project(r):
    return (f"ftune_{STAGE}_{r['arm']}_s{r['steps']}_c{r['wcov']}_u{r['wctrl']}"
            f"__m{r['m']}_nm{r['nm']}_seed{r['seed']}")


def trained(r):
    f = os.path.join(LOG, project(r), "koopman_results_log.csv")
    return os.path.exists(f) and len(pd.read_csv(f)) > 0


def train(r, gpu):
    flags = ["--use_residual", "--multiply_encode_by_input_dim"]
    if r["arm"] in ("cov", "both"):
        flags.append("--use_covariance_loss")
    if r["arm"] in ("ctrl", "both"):
        flags.append("--use_control_loss")
    cmd = [PY, "train_model.py", "--project_name", project(r), "--env_name", "FrankaImproved",
           "--sample_size", str(r["m"]), "--encode_dim", str(r["nm"]), "--layer_depth", "3", "--hidden_dim", "256",
           "--seed", str(r["seed"]), "--train_steps", str(r["steps"]), "--cov_loss_weight", str(r["wcov"]),
           "--ctrl_loss_weight", str(r["wctrl"]), "--log_name", "koopman_results_log.csv", "--device", "cuda",
           "--use_compile", ""] + flags
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "4"))
    with open(os.path.join(LOG, project(r) + ".train.txt"), "w") as fh:
        return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)


def evaluate(args):
    path, encode_dim = args
    sys.path.insert(0, os.path.join(HERE, "..", "control"))
    sys.path.insert(0, os.path.join(HERE, "..", "utility"))
    os.chdir(os.path.join(HERE, "..", "control"))
    import numpy as np
    import franka_mpc as fm
    np.random.seed(0)
    res = fm.evaluate_franka_mpc(path, int(encode_dim), fm.FrankaMPCConfig(warmup_steps=50), device="cpu")
    return path, res.error_figure8, res.error_star, res.max_eigenvalue


def main():
    os.makedirs(os.path.join(LOG, "franka_tuning"), exist_ok=True)
    todo = runs()
    if os.environ.get("SMOKE"):   # quick end-to-end check: two short runs
        todo = [dict(r, steps=2000) for r in todo[:2]]
    print(f"[ftune] stage={STAGE} shard={SHARD}/{NSHARDS}: {len(todo)} runs", flush=True)
    procs, k = [], 0
    for r in todo:
        if trained(r):
            continue
        while len([p for p in procs if p.poll() is None]) >= PARALLEL:
            time.sleep(5)
        procs.append(train(r, k % NGPU)); k += 1
        print(f"[ftune] train {project(r)}", flush=True)
    for p in procs:
        p.wait()
    rows = []
    for r in todo:
        f = os.path.join(LOG, project(r), "koopman_results_log.csv")
        if not os.path.exists(f):
            print(f"[ftune] MISSING training log for {project(r)}", flush=True)
            continue
        d = pd.read_csv(f).iloc[-1]
        rows.append(dict(r, project=project(r), model_path=d["model_path"], encode_dim=int(d["encode_dim"]),
                         test_Kloss=d["test_Kloss"]))
    models = pd.DataFrame(rows)
    out = os.path.join(LOG, "franka_tuning", f"{STAGE}_results_{SHARD}.csv")
    done = set(pd.read_csv(out).model_path) if os.path.exists(out) else set()
    pending = [(p, e) for p, e in zip(models.model_path, models.encode_dim) if p not in done]
    print(f"[ftune] evaluating {len(pending)} models ({len(done)} already done)", flush=True)
    with ProcessPoolExecutor(EVAL_PAR) as ex:
        for path, fig8, star, rho in ex.map(evaluate, pending):
            row = models[models.model_path == path].iloc[0].to_dict()
            row.update(error_figure8=fig8, error_star=star, max_eigenvalue=rho)
            df = pd.DataFrame([row])
            df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"[ftune] {row['project']}: fig8 {fig8:.4f} star {star:.4f}", flush=True)


if __name__ == "__main__":
    main()
