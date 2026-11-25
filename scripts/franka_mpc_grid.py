"""Franka MPC-setting study: one grid of linear-MPC settings, the same for every model type.

Trains the models of the Franka final grid (franka_tune.py, STAGE=final settings: Baseline, +Cov, +Ctrl, +Aux on
m in {1, 4, 16, 64, 140}k x n_mult in {1, 2, 4, 8, 16} x 5 seeds), then evaluates every model (and the EDMD
baselines in data/edmd_franka/*.pt) with each MPC setting of CONFIGS on the star and figure-8 trajectories.
Settings are selected later on the star trajectory only (analyse_franka_mpc_grid.py); figure-8 is the test.

The controller is the unconstrained linear MPC of franka_mpc.FrankaMPC (closed-form solution, then clipping),
restricted to the rows that carry cost (the first 14 lifted coordinates are the normalized state).
Grid: horizon H, input weight R (on normalized inputs), and the joint-velocity reference:
  zero : reference velocity 0 (the original controller), weight Q_vel
  fd   : reference velocity from finite differences of the IK joint reference, weight Q_vel
The original controller is H=15, R=0, (zero, Q_vel=1).

MODE=bothfinal trains the selected +Aux setting (BOTH_WCOV, BOTH_WCTRL, BOTH_STEPS) on the full grid.
MODE=bothsel instead trains a wider +Aux sweep for selection (m = 140k, n_mult in {1, 4, 16}, two seeds):
w_cov in {0.01, 0.1, 1} x w_ctrl in {0.1, 1, 3} x train_steps in {120k, 240k}; the loss setting and the MPC
setting are then selected jointly on the star trajectory (projects ftune_bothsel_*).

Env: MODE=final|bothsel, SHARD, NSHARDS, NGPU, PARALLEL, EVAL_PAR, SKIP_TRAIN=1 (evaluate only), SMOKE=1.
"""
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

MODE = os.environ.get("MODE", "final")
os.environ["STAGE"] = MODE
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import franka_tune as ft  # noqa: E402

LOG = ft.LOG
OUT_DIR = os.path.join(LOG, "franka_mpc_grid" if os.environ.get("MODE", "final") == "final" else "franka_mpc_" + os.environ["MODE"])
CONFIGS = [dict(H=H, R=R, vel=vel, Qv=Qv)
           for H in (5, 10, 15, 25)
           for R in (0.0, 1e-3, 1e-2, 1e-1, 1.0)
           for vel, Qv in (("zero", 1.0), ("zero", 0.0), ("fd", 1.0), ("fd", 0.1))]
Q_POS, STEPS, WARMUP, NOISE, DT, CLIP, RHO_MAX = 10.0, 300, 50, 0.01, 0.02, 0.3, 1.10
H_MAX = max(c["H"] for c in CONFIGS)

_ENV = {}


def _fm():
    sys.path.insert(0, os.path.join(HERE, "..", "control"))
    sys.path.insert(0, os.path.join(HERE, "..", "control", "baseline_control"))
    sys.path.insert(0, os.path.join(HERE, "..", "utility"))
    import franka_mpc as fm
    return fm


def references():
    """IK joint references, computed once per process exactly as franka_mpc.evaluate_franka_mpc does."""
    if "refs" in _ENV:
        return _ENV["env"], _ENV["refs"]
    fm = _fm()
    os.chdir(os.path.join(HERE, "..", "control"))
    env = fm.FrankaEnv(render=False)
    refs = {}
    for name, gen, T0 in (("figure8", fm.generate_figure8_trajectory, STEPS + H_MAX + 1),
                          ("star", fm.generate_star_trajectory, STEPS + 15 + 1)):   # star shape depends on T
        pos = gen(T0, DT)
        pos = np.vstack([pos, np.repeat(pos[-1:], STEPS + H_MAX + 1 - len(pos), axis=0)])
        env.reset()
        q = np.array([fm.ik_to(env, *p) for p in pos])
        q_init = np.array(fm.ik_to(env, pos[0, 0], pos[0, 1] + 0.01, pos[0, 2] + 0.043))
        dq = np.vstack([np.diff(q, axis=0) / DT, np.zeros((1, 7))])
        dq[-1] = dq[-2]
        refs[name] = dict(pos=pos, q=q, dq=dq, q_init=q_init)
    _ENV.update(env=env, refs=refs)
    return env, refs


class Model:
    """encode(x) -> lifted z (first 14 entries = normalized state), z+ = A z + B u_norm."""

    def __init__(self, kind, path, encode_dim=None):
        fm = _fm()
        import torch
        torch.set_num_threads(1)
        if kind == "koopman":
            net, A, B, ns = fm.load_koopman_model(path, 14, 7, int(encode_dim), device="cpu")
            self.net, self.torch = net, torch

            def enc(x):
                with torch.inference_mode():
                    return net.encode(torch.tensor(x.reshape(1, -1), dtype=torch.float64)).numpy().ravel()
            self._enc = enc
        else:
            from baseline_mpc_utils import load_edmd_from_checkpoint
            edmd, ns = load_edmd_from_checkpoint(path)
            A, B = edmd.A, edmd.B
            self._enc = lambda x: edmd.lift(x.reshape(1, -1)).ravel()
        self.A, self.B, self.ns = np.asarray(A, float), np.asarray(B, float), ns
        self.rho = float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def norm(self, x):
        if self.ns is None:
            return x
        return (x - np.asarray(self.ns["state_mean"])) / np.asarray(self.ns["state_std"])

    def encode(self, x):
        return self._enc(self.norm(x))

    def denorm_u(self, u):
        if self.ns is None:
            return u
        return u * np.asarray(self.ns["action_std"]) + np.asarray(self.ns["action_mean"])


def gains(model, H, R, Qv):
    """u_norm = -(K_z z - K_r r): first input of the closed-form MPC, with r the stacked normalized reference."""
    A, B, n, mu = model.A, model.B, model.A.shape[0], model.B.shape[1]
    S = 14
    Ap = [np.eye(n)]
    for _ in range(H):
        Ap.append(A @ Ap[-1])
    Mz = np.vstack([Ap[k][:S] for k in range(H + 1)])                      # ((H+1)S, n)
    C = np.zeros(((H + 1) * S, H * mu))
    for i in range(1, H + 1):
        for j in range(i):
            C[i * S:(i + 1) * S, j * mu:(j + 1) * mu] = (Ap[i - j - 1] @ B)[:S]
    qd = np.tile(np.r_[np.full(7, Q_POS), np.full(7, Qv)], H + 1)
    P = R * np.eye(H * mu) + C.T @ (qd[:, None] * C)
    G = np.linalg.lstsq(P, C.T * qd, rcond=None)[0][:mu]    # (mu, (H+1)S); minimum-norm minimizer if P is singular
    return G @ Mz, G


def track(model, traj, cfg, seed):
    env, refs = references()
    import pybullet as pb
    fm = _fm()
    r = refs[traj]
    H = cfg["H"]
    x_ref = np.hstack([r["q"], r["dq"] if cfg["vel"] == "fd" else np.zeros_like(r["dq"])])
    xr = model.norm(x_ref)                                                   # (T, 14) normalized reference
    Kz, Kr = gains(model, H, cfg["R"], cfg["Qv"])
    rng = np.random.default_rng(seed)
    env.reset()
    for j, v in enumerate(r["q_init"]):
        pb.resetJointState(env.robot, j, float(v))
    full = env.get_state()
    errs = []
    for step in range(STEPS):
        x = fm.extract_joint_state(full).copy()
        x[:7] += rng.standard_normal(7) * NOISE
        z = model.encode(x)
        u = model.denorm_u(-(Kz @ z - Kr @ xr[step:step + H + 1].ravel()))
        if not np.all(np.isfinite(u)):
            return float("inf")
        full = env.step(np.clip(u, -CLIP, CLIP))
        if step >= WARMUP:
            errs.append(np.linalg.norm(fm.get_ee_position(env)[1:3] - r["pos"][step, 1:3]))
    return float(np.mean(errs))


def evaluate(item):
    kind, path, encode_dim, meta = item
    out = []
    try:
        model = Model(kind, path, encode_dim)
    except Exception as e:                       # noqa: BLE001  (record and continue)
        print(f"[grid] load failed {path}: {e}", flush=True)
        return out
    for ci, cfg in enumerate(CONFIGS):
        row = dict(meta, kind=kind, model_path=path, rho=model.rho, cfg=ci, **cfg)
        if model.rho > RHO_MAX:
            row.update(err_star=float("inf"), err_figure8=float("inf"))
        else:
            try:
                row.update(err_star=track(model, "star", cfg, 1), err_figure8=track(model, "figure8", cfg, 2))
            except (np.linalg.LinAlgError, ValueError) as e:      # record the failure, keep the other settings
                print(f"[grid] {path} cfg {ci}: {e}", flush=True)
                row.update(err_star=float("inf"), err_figure8=float("inf"))
        out.append(row)
    return out


def runs():
    if MODE == "bothfinal":     # the +Aux setting selected on the star trajectory, on the full grid
        c, u, st = float(os.environ["BOTH_WCOV"]), float(os.environ["BOTH_WCTRL"]), int(os.environ["BOTH_STEPS"])
        out = [dict(arm="both", steps=st, wcov=c, wctrl=u, m=m, nm=nm, seed=seed)
               for m in (1000, 4000, 16000, 64000, 140000) for nm in (1, 2, 4, 8, 16) for seed in ft.SEEDS5]
        return [r for i, r in enumerate(out) if i % ft.NSHARDS == ft.SHARD]
    if MODE != "bothsel":
        return ft.runs()
    out = [dict(arm="both", steps=st, wcov=c, wctrl=u, m=140000, nm=nm, seed=seed)
           for st in (120000, 240000) for c in (0.01, 0.1, 1.0) for u in (0.1, 1.0, 3.0)
           for nm in (1, 4, 16) for seed in ft.SEEDS5[:2]]
    return [r for i, r in enumerate(out) if i % ft.NSHARDS == ft.SHARD]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = runs()
    if os.environ.get("SMOKE"):
        todo = [dict(r, steps=2000) for r in todo[:2]]
        CONFIGS[:] = [c for c in CONFIGS if c["R"] in (0.0, 1e-2) and c["H"] in (10, 15)]
    if not os.environ.get("SKIP_TRAIN"):
        procs, k = [], 0
        for r in todo:
            if ft.trained(r):
                continue
            while len([p for p in procs if p.poll() is None]) >= ft.PARALLEL:
                time.sleep(5)
            procs.append(ft.train(r, k % ft.NGPU)); k += 1
            print(f"[grid] train {ft.project(r)}", flush=True)
        for p in procs:
            p.wait()
    items = []
    for r in todo:
        f = os.path.join(LOG, ft.project(r), "koopman_results_log.csv")
        if not os.path.exists(f):
            print(f"[grid] MISSING {ft.project(r)}", flush=True)
            continue
        d = pd.read_csv(f).iloc[-1]
        items.append(("koopman", d["model_path"], int(d["encode_dim"]),
                      dict(arm=r["arm"], m=r["m"], nm=r["nm"], seed=r["seed"], test_Kloss=d["test_Kloss"],
                           steps=r["steps"], wcov=r["wcov"], wctrl=r["wctrl"])))
    if ft.SHARD == 0 and MODE == "final":
        for p in sorted(glob.glob(os.path.join(HERE, "..", "data", "edmd_franka", "edmd_*.pt"))):
            _, _, seed, nm, m = os.path.basename(p)[:-3].split("_")
            items.append(("edmd", p, None, dict(arm="edmd", m=int(m), nm=int(nm), seed=int(seed), test_Kloss=np.nan)))
    print(f"[grid] evaluating {len(items)} models x {len(CONFIGS)} settings", flush=True)
    rows = []
    with ProcessPoolExecutor(ft.EVAL_PAR) as ex:
        for i, res in enumerate(ex.map(evaluate, items)):
            rows += res
            if i % 10 == 0:
                print(f"[grid] {i + 1}/{len(items)} models done", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"grid_{ft.SHARD}.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, f"grid_{ft.SHARD}.csv"), index=False)
    print("[grid] done", flush=True)


if __name__ == "__main__":
    main()
