# =========================
# Experiments to run (filter lists)
# =========================
project_name = "Aug_8"
envs        = ["DampingPendulum", "DoublePendulum"]
encode_dims = [1, 2, 4, 8, 16]
cov_regs    = [0, 1]
ctrl_regs   = [0, 1]
train_sizes = [1000, 4000, 16000, 60000]
seeds       = [17382, 76849, 20965, 84902, 51194]

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.append("../utility")
from network import KoopmanNet
from Utility import data_collecter
import lqr

# ======================================================
# Tunables for goal detection & evaluation
# ======================================================
EVAL_DT_DP          = 0.02  # DampingPendulum
EVAL_DT_DBL         = 0.01  # DoublePendulum
LASTN_ERR_STEPS     = 10    # last-N window for last10_L1
EVAL_STEPS_DP       = 200   # rollout length for DampingPendulum
EVAL_STEPS_DBL      = 200   # rollout length for DoublePendulum
STABILITY_RADIUS    = 1.10  # spectral radius threshold for closed-loop Acl
USE_RESIDUAL_NET    = True
ACTION_CLIP         = None  # e.g., (-5, 5) or None

# Goal tube (angles wrapped)
GOAL_ANGLE_TOL      = 0.05  # radians
GOAL_VEL_TOL        = 0.10  # rad/s
GOAL_DWELL_STEPS    = 50    # consecutive steps to declare capture

# ======================================================
# Angle / velocity indexing
# ======================================================
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi  # (-pi, pi]

def get_angle_indices(env_name):
    if env_name.startswith("DampingPendulum"):
        return [0]          # theta
    elif env_name.startswith("DoublePendulum"):
        return [0, 1]       # theta1, theta2
    return []

def get_velocity_indices(env_name):
    if env_name.startswith("DampingPendulum"):
        return [1]          # theta_dot
    elif env_name.startswith("DoublePendulum"):
        return [2, 3]       # theta1_dot, theta2_dot
    return []

def wrap_state_error(err, angle_idx):
    err_wrapped = err.copy()
    for i in angle_idx:
        err_wrapped[i, :] = np.vectorize(wrap_angle)(err_wrapped[i, :])
    return err_wrapped

# ======================================================
# Helpers
# ======================================================
def resolve_dt(env_name):
    """Use explicit per-env dt (overrides any env.dt) for TTG calculation."""
    if env_name.startswith("DoublePendulum"):
        return EVAL_DT_DBL
    if env_name.startswith("DampingPendulum"):
        return EVAL_DT_DP

def spectral_radius(A):
    vals = np.linalg.eigvals(np.asarray(A))
    return float(np.max(np.abs(vals)))

def Psi_o(s, net, NKoopman):
    """Encoded observable (no control features); returns column vector (NKoopman,1)."""
    psi = np.zeros([NKoopman, 1])
    ds = net.encode(torch.as_tensor(s, dtype=torch.double)).detach().cpu().numpy()
    psi[:NKoopman, 0] = ds
    return psi

def prepare_refs(env_name, Nstate):
    x_ref = np.zeros(Nstate)
    if env_name.startswith("DampingPendulum"):
        reset_state = [-2.5, 0.1]
    elif env_name.startswith("DoublePendulum"):
        x_ref[0], x_ref[1] = 1.2, -0.8
        reset_state = [0, 0, 0, 0]
    else:
        raise ValueError(f"Unknown env_name {env_name}")
    return reset_state, x_ref

def build_Q(env_name, NKoopman, Nstate, angle_w, vel_w, latent_w):
    Q = np.zeros((NKoopman, NKoopman))
    angle_idx = get_angle_indices(env_name)
    vel_idx   = get_velocity_indices(env_name)
    for i in angle_idx: Q[i, i] = angle_w
    for i in vel_idx:   Q[i, i] = vel_w
    if latent_w > 0:
        Q[Nstate:, Nstate:] = latent_w * np.eye(NKoopman - Nstate)
    return np.matrix(Q)

def build_R(udim, scale):
    return np.matrix(scale * np.eye(udim))

def Done(env_name, state):
    # Early termination if state "blows up"
    if env_name.startswith("DampingPendulum"):
        return abs(state[0]) >= 2 * np.pi
    elif env_name.startswith("DoublePendulum"):
        return abs(state[0]) >= 3 * np.pi or abs(state[1]) >= 3 * np.pi
    return False

def to_bool_list(vals):
    """Normalize a list like [0,1] or [False,True] to booleans."""
    out = []
    for v in vals:
        if isinstance(v, (np.bool_, bool)):
            out.append(bool(v))
        elif isinstance(v, (int, np.integer)):
            out.append(bool(int(v)))
        elif isinstance(v, str):
            out.append(v.strip().lower() in ("1","true","t","yes","y"))
        else:
            out.append(bool(v))
    # keep both orders allowed in logs; distinct set
    return [False, True] if set(out) == {False, True} else sorted(list(set(out)))

# ======================================================
# Metrics (existing)
# ======================================================
def compute_tracking_metrics(observations, x_ref, angle_idx, last_N=10):
    T = observations.shape[1]
    err = observations - x_ref.reshape(-1, 1)
    err = wrap_state_error(err, angle_idx)
    N_tail = min(last_N, T)
    err_lastN_L1 = float(np.mean(np.abs(err[:, -N_tail:])))
    err_full_L1  = float(np.mean(np.abs(err)))
    return err_lastN_L1, err_full_L1

def cost_with_wrap(observations, u_list, Q_state, R_eval, x_ref, angle_idx):
    T = observations.shape[1]
    loss = 0.0
    x_ref_col = x_ref.reshape(-1, 1)
    for s in range(T):
        e = wrap_state_error(observations[:, s:s+1] - x_ref_col, angle_idx)
        loss += np.asarray(e.T @ Q_state @ e).item()
        if s < T - 1 and len(u_list) > 0:
            u = u_list[s].reshape(-1, 1)
            loss += np.asarray(u.T @ R_eval @ u).item()
    return float(loss)

# ======================================================
# Goal detection helpers (new)
# ======================================================
def reached_goal_flags(observations, x_ref, angle_idx, vel_idx, angle_tol, vel_tol):
    """Boolean per-timestep: within goal tube (angles wrapped)."""
    err = observations - x_ref.reshape(-1, 1)
    err = wrap_state_error(err, angle_idx)
    T = err.shape[1]
    if len(angle_idx):
        ok_angle = np.all(np.abs(err[angle_idx, :]) <= angle_tol, axis=0)
    else:
        ok_angle = np.ones(T, dtype=bool)
    if len(vel_idx):
        ok_vel   = np.all(np.abs(err[vel_idx,   :]) <= vel_tol,   axis=0)
    else:
        ok_vel   = np.ones(T, dtype=bool)
    return ok_angle & ok_vel

def goal_captured(ok_flags, dwell_steps):
    """Return (captured: bool, first_hit_idx: int | None)."""
    run = 0
    for i, ok in enumerate(ok_flags):
        run = run + 1 if ok else 0
        if run >= dwell_steps:
            return True, i - dwell_steps + 1
    return False, None

def time_to_goal_seconds(first_hit_idx, dt):
    return float(first_hit_idx * dt) if first_hit_idx is not None else float('inf')

# ======================================================
# Grids
# ======================================================
DAMPING_GRID = {
    "uval":     [1e-3, 1e-2, 1e-1],
    "angle_w":  [1e-1, 1.0, 10.0],
    "vel_w":    [1e-1, 1.0],
    "latent_w": [0.0],
}

DOUBLE_GRID = {
    "uval":     [1e-3, 1e-2, 1e-1],
    "angle_w":  [1e-1, 1.0, 10.0],
    "vel_w":    [1e-1, 1.0],
    "latent_w": [0.0],
}

def grid_for_env(env_name):
    return DOUBLE_GRID if env_name.startswith("DoublePendulum") else DAMPING_GRID

def ordered_uvals(env_name, ctrl_loss):
    g = grid_for_env(env_name)
    uvals = list(g["uval"])
    uvals.sort()
    return uvals

# ======================================================
# Single rollout for specified params
# params = (uval, angle_w, vel_w, latent_w)
# ======================================================
def evaluate_model_once(model_path, env_name, encode_dim, params,
                        steps=200, lastN=10, stability_rad=1.05,
                        use_residual=True, action_clip=None):
    uval, angle_w, vel_w, latent_w = params

    Data_collect = data_collecter(env_name)
    udim, Nstate = Data_collect.udim, Data_collect.Nstates

    # Robust torch.load for PyTorch>=2.6 (weights_only default True)
    subsuffix = "../" + model_path[3:]
    try:
        checkpoint = torch.load(subsuffix, map_location=torch.device('cpu'), weights_only=False)
    except TypeError:
        checkpoint = torch.load(subsuffix, map_location=torch.device('cpu'))
    state_dict, Elayer = checkpoint["model"], checkpoint["layer"]

    NKoopman = encode_dim + Nstate
    net = KoopmanNet(Elayer, NKoopman, udim, use_residual)
    net.load_state_dict(state_dict); net.cpu(); net.double()

    Ad = np.matrix(state_dict['lA.weight'].cpu().numpy())
    Bd = np.matrix(state_dict['lB.weight'].cpu().numpy())

    reset_state, x_ref = prepare_refs(env_name, Nstate)
    Q_full = build_Q(env_name, NKoopman, Nstate, angle_w, vel_w, latent_w)
    R = build_R(udim, 1.0)  # base R; scaled by uval in LQR

    # LQR gain
    try:
        Kopt = lqr.lqr_regulator_k(Ad, Bd, Q_full, uval * R)
    except Exception:
        return None  # invalid Riccati or dims

    # Early stability check
    Acl = Ad - Bd @ Kopt
    if spectral_radius(Acl) > stability_rad:
        return None

    env = Data_collect.env
    env.reset()
    angle_idx = get_angle_indices(env_name)
    vel_idx   = get_velocity_indices(env_name)

    observation_list, u_list = [], []
    observation = env.reset_state(reset_state)
    x0 = np.matrix(Psi_o(observation, net, NKoopman))
    x_ref_lift = Psi_o(x_ref, net, NKoopman)
    observation_list.append(np.array(x0[:Nstate]).reshape(-1, 1))

    failed = False
    for _ in range(steps):
        u = -Kopt @ (x0 - x_ref_lift)
        u_np = np.asarray(u).flatten()
        if action_clip is not None:
            u_np = np.clip(u_np, action_clip[0], action_clip[1])
        action = u_np[0] if u_np.size == 1 else u_np
        observation, _, _, _ = env.step(action)
        if Done(env_name, observation):
            failed = True
            break  # compute metrics on what we have so far
        x0 = np.matrix(Psi_o(observation, net, NKoopman))
        observation_list.append(np.array(x0[:Nstate]).reshape(-1, 1))
        u_list.append(u_np.astype(float))

    observations = np.concatenate(observation_list, axis=1)  # (Nstate, T)

    # Costs & errors
    Q_state = np.array(Q_full)[:Nstate, :Nstate]
    R_eval  = uval * np.array(R)
    last10_L1, full_L1 = compute_tracking_metrics(observations, x_ref, angle_idx, last_N=lastN)
    J = cost_with_wrap(observations, u_list, Q_state, R_eval, x_ref, angle_idx)

    # Goal metrics
    dt = resolve_dt(env_name)
    ok_flags = reached_goal_flags(
        observations, x_ref, angle_idx, vel_idx,
        angle_tol=GOAL_ANGLE_TOL, vel_tol=GOAL_VEL_TOL
    )
    captured, first_hit_idx = goal_captured(ok_flags, dwell_steps=GOAL_DWELL_STEPS)
    ttg = time_to_goal_seconds(first_hit_idx, dt)

    return {
        "goal_reached": int(captured),
        "time_to_goal_sec": ttg,
        "last10_L1": last10_L1,
        "full_L1": full_L1,
        "control_cost": J,
        "failed_early": int(failed),
        "params": {"uval": uval, "angle_w": angle_w, "vel_w": vel_w, "latent_w": latent_w}
    }

# ======================================================
# Single-stage grid search with goal-based selection
# ======================================================
def _better_candidate(a, b):
    """Return True if a is better than b under the selection order."""
    if b is None:
        return True
    # 1) goal_reached: prefer 1 over 0
    if a["goal_reached"] != b["goal_reached"]:
        return a["goal_reached"] > b["goal_reached"]
    # 2) time to goal: smaller is better
    if a["time_to_goal_sec"] != b["time_to_goal_sec"]:
        return a["time_to_goal_sec"] < b["time_to_goal_sec"]
    # 3) last10_L1: smaller is better
    if a["last10_L1"] != b["last10_L1"]:
        return a["last10_L1"] < b["last10_L1"]
    # 4) control cost: smaller is better
    return a["control_cost"] < b["control_cost"]

def evaluate_model_grid(model_path, env_name, encode_dim, ctrl_loss,
                        steps=150, lastN=10, stability_rad=1.05,
                        action_clip=None, use_residual=True):
    grid = grid_for_env(env_name)
    uvals = ordered_uvals(env_name, ctrl_loss)

    best = None
    tried = 0

    for uval in uvals:
        for angle_w in grid["angle_w"]:
            for vel_w in grid["vel_w"]:
                for latent_w in grid["latent_w"]:
                    tried += 1
                    out = evaluate_model_once(
                        model_path, env_name, encode_dim,
                        (uval, angle_w, vel_w, latent_w),
                        steps=steps, lastN=lastN, stability_rad=stability_rad,
                        use_residual=use_residual, action_clip=action_clip
                    )
                    if out is None:
                        continue
                    if _better_candidate(out, best):
                        best = out

    return best, tried

# ======================================================
# Utilities
# ======================================================
def _safe_train_samples(val):
    try:
        if pd.isna(val):
            return np.nan
        iv = int(val)
        return iv
    except Exception:
        return val

def _median_safe(series_like):
    x = np.asarray(series_like, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size > 0 else np.nan

def _mean_safe(series_like):
    x = np.asarray(series_like, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size > 0 else np.nan

# ======================================================
# Main
# ======================================================
def main():
    base_dir = f"../log/{project_name}"
    os.makedirs(base_dir, exist_ok=True)

    log_path = f"{base_dir}/koopman_results_log.csv"
    raw = pd.read_csv(log_path)

    # -------- Map encode_dim multipliers -> absolute per ENV --------
    # Get Nstates for each env you plan to include
    env_dims = {}
    for e in envs:
        try:
            dc = data_collecter(e)
            env_dims[e] = int(dc.Nstates)
        except Exception:
            # sensible defaults if constructor fails
            env_dims[e] = 2 if e.startswith("DampingPendulum") else 4

    # Build per-env allowed absolute encode_dim sets
    enc_abs_per_env = {
        e: sorted({m * env_dims[e] for m in encode_dims}) for e in envs
    }

    # -------- Normalize boolean filters --------
    cov_bools  = to_bool_list(cov_regs)
    ctrl_bools = to_bool_list(ctrl_regs)

    # -------- Column checks --------
    need = {'env_name','encode_dim','use_covariance_loss','use_control_loss','seed','model_path','train_samples'}
    missing = sorted(need - set(raw.columns))
    if missing:
        raise KeyError(f"Log is missing required columns for filtering: {missing}")

    # Ensure numeric train_samples for filtering
    raw['train_samples'] = pd.to_numeric(raw['train_samples'], errors='coerce')

    # -------- Apply filters (env + per-env encode_dim + regs + seeds + train_samples) --------
    def row_ok(r):
        if r['env_name'] not in envs:
            return False
        enc_ok = int(r['encode_dim']) in enc_abs_per_env[r['env_name']]
        cov_ok = bool(r['use_covariance_loss']) in cov_bools
        ctl_ok = bool(r['use_control_loss']) in ctrl_bools
        seed_ok = int(r['seed']) in seeds
        ts_ok = (not np.isnan(r['train_samples'])) and (int(r['train_samples']) in train_sizes)
        return enc_ok and cov_ok and ctl_ok and seed_ok and ts_ok

    log = raw[raw.apply(row_ok, axis=1)].copy().reset_index(drop=True)

    if len(log) == 0:
        print("[Error] No models found after filtering.")
        def show_unique(col):
            return sorted(raw[col].dropna().unique().tolist()) if col in raw.columns else "N/A"
        print("Requested envs:", envs)
        print("Per-env encode_dim (multipliers -> absolute):", enc_abs_per_env)
        print("Available encode_dim:", show_unique('encode_dim'))
        print("Requested cov_regs ->", cov_regs, "normalized to", cov_bools, "; available:", show_unique('use_covariance_loss'))
        print("Requested ctrl_regs ->", ctrl_regs, "normalized to", ctrl_bools, "; available:", show_unique('use_control_loss'))
        print("Requested seeds:", seeds, "; available:", show_unique('seed'))
        print("Requested train_samples:", train_sizes, "; available:", show_unique('train_samples'))
        return

    rows = []
    for i in range(len(log)):
        row = log.iloc[i]
        model_path    = row['model_path']
        env_name      = row['env_name']
        encode_dim    = int(row['encode_dim'])              # absolute latent size
        ctrl_loss     = row.get('use_control_loss', None)
        cov_loss      = row.get('use_covariance_loss', None)
        seed          = row.get('seed', None)
        train_samples = _safe_train_samples(row['train_samples'])

        try:
            if env_name.startswith("DoublePendulum"):
                sr    = STABILITY_RADIUS
                clip  = ACTION_CLIP
                steps = EVAL_STEPS_DBL
            else:
                sr    = STABILITY_RADIUS
                clip  = ACTION_CLIP
                steps = EVAL_STEPS_DP

            best, tried = evaluate_model_grid(
                model_path=model_path,
                env_name=env_name,
                encode_dim=encode_dim,
                ctrl_loss=ctrl_loss,
                steps=steps,
                lastN=LASTN_ERR_STEPS,
                stability_rad=sr,
                action_clip=clip,
                use_residual=USE_RESIDUAL_NET
            )

            total_tried = tried
            if best is None:
                print(
                    f"❌ Failure: {env_name} | enc={encode_dim} | ctrl_loss={ctrl_loss} | "
                    f"cov_loss={cov_loss} | seed={seed} | train_samples={train_samples} | "
                    f"(no stable candidate out of {total_tried})"
                )
                rows.append({
                    'env_name': env_name, 'encode_dim': encode_dim,
                    'use_control_loss': ctrl_loss, 'use_covariance_loss': cov_loss,
                    'seed': seed, 'train_samples': train_samples,
                    'goal_reached': 0,
                    'time_to_goal_sec': np.inf,
                    'best_last10_L1': np.inf,
                    'best_full_L1': np.inf,
                    'best_control_cost': np.inf,
                    'best_uval': None, 'best_angle_w': None,
                    'best_vel_w': None, 'best_latent_w': None,
                    'tried_total': total_tried,
                    'model_path': model_path,
                    'stable': 0
                })
            else:
                p = best["params"]
                print(
                    f"✅ Best: {env_name} | enc={encode_dim} | ctrl_loss={ctrl_loss} | "
                    f"cov_loss={cov_loss} | seed={seed} | train_samples={train_samples} | "
                    f"goal_reached={best['goal_reached']} | ttg={best['time_to_goal_sec']:.3f}s | "
                    f"last10_L1={best['last10_L1']:.4f} | J={best['control_cost']:.2f} "
                    f"(from total {total_tried})"
                )
                rows.append({
                    'env_name': env_name, 'encode_dim': encode_dim,
                    'use_control_loss': ctrl_loss, 'use_covariance_loss': cov_loss,
                    'seed': seed, 'train_samples': train_samples,
                    'goal_reached': best['goal_reached'],
                    'time_to_goal_sec': best['time_to_goal_sec'],
                    'best_last10_L1': best['last10_L1'],
                    'best_full_L1': best['full_L1'],
                    'best_control_cost': best['control_cost'],
                    'best_uval': p['uval'], 'best_angle_w': p['angle_w'],
                    'best_vel_w': p['vel_w'], 'best_latent_w': p['latent_w'],
                    'tried_total': total_tried,
                    'model_path': model_path,
                    'stable': 1
                })

        except Exception as e:
            print(
                f"❌ Failed: {env_name} | enc={encode_dim} | ctrl_loss={ctrl_loss} | "
                f"cov_loss={cov_loss} | seed={seed} | train_samples={train_samples} | Error: {e}"
            )
            rows.append({
                'env_name': env_name, 'encode_dim': encode_dim,
                'use_control_loss': ctrl_loss, 'use_covariance_loss': cov_loss,
                'seed': seed, 'train_samples': train_samples,
                'goal_reached': 0,
                'time_to_goal_sec': np.inf,
                'best_last10_L1': np.inf, 'best_full_L1': np.inf, 'best_control_cost': np.inf,
                'best_uval': None, 'best_angle_w': None, 'best_vel_w': None, 'best_latent_w': None,
                'tried_total': 0, 'model_path': model_path, 'stable': 0
            })

    # -------------------------
    # Save per-model BEST results
    # -------------------------
    df_best = pd.DataFrame(rows)
    if 'train_samples' in df_best.columns:
        df_best['train_samples'] = pd.to_numeric(df_best['train_samples'], errors='coerce')

    best_path = f"{base_dir}/pendulum_control_results_best.csv"
    df_best.to_csv(best_path, index=False)
    print(f"\nSaved BEST-per-model results to: {best_path}")

    # -------------------------
    # Grouped summary (primary: goal metrics; keep error & J)
    # -------------------------
    group_keys = ['env_name', 'encode_dim', 'use_control_loss', 'use_covariance_loss', 'train_samples']
    grouped = df_best.groupby(group_keys, dropna=False)

    summary_rows = []
    for cfg, g in grouped:
        total = len(g)
        stable_cnt = int(g['stable'].sum())
        stability_rate = stable_cnt / total if total > 0 else np.nan

        # Primary metrics
        success_rate = _mean_safe(g['goal_reached'])  # proportion
        # median over finite ttg only
        ttg_finite = g['time_to_goal_sec'].replace([np.inf, -np.inf], np.nan)
        median_ttg = _median_safe(ttg_finite)

        # Keep error & cost summaries
        med_last10 = _median_safe(g['best_last10_L1'])
        med_J      = _median_safe(g['best_control_cost'])

        summary_rows.append({
            'env_name': cfg[0],
            'encode_dim': cfg[1],
            'use_control_loss': cfg[2],
            'use_covariance_loss': cfg[3],
            'train_samples': cfg[4],
            'n_models': total,
            'n_stable': stable_cnt,
            'stability_rate': stability_rate,
            'success_rate': success_rate,
            'median_time_to_goal_sec': median_ttg,
            'median_best_last10_L1': med_last10,
            'median_best_control_cost': med_J,
        })

    df_summary = pd.DataFrame(summary_rows).sort_values(group_keys).reset_index(drop=True)

    summary_path = f"{base_dir}/pendulum_control_results.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved grouped SUMMARY (goal metrics + last10 & cost) to: {summary_path}")

if __name__ == "__main__":
    main()