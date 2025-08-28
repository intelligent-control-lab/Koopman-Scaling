import os
import numpy as np
import pandas as pd
import torch

import sys
sys.path.append("../../utility")
from network import KoopmanNet
from dataset import KoopmanDatasetCollector
import lqr

# ============================
# Angle / velocity indexing
# ============================
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

# ============================
# Helpers
# ============================
def spectral_radius(A):
    vals = np.linalg.eigvals(np.asarray(A))
    return float(np.max(np.abs(vals)))

def Psi_o(s, net, NKoopman):
    psi = np.zeros([NKoopman, 1])
    ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
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
    if env_name.startswith("DampingPendulum"):
        return abs(state[0]) >= 2 * np.pi
    elif env_name.startswith("DoublePendulum"):
        return abs(state[0]) >= 3 * np.pi or abs(state[1]) >= 3 * np.pi
    return False

# ============================
# Metrics
# ============================
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

# ============================
# Single rollout for specified params
# params = (uval, angle_w, vel_w, latent_w)
# ============================
def evaluate_model_once(model_path, env_name, encode_dim, params,
                        steps=200, lastN=10, stability_rad=1.05,
                        use_residual=True, action_clip=None):
    uval, angle_w, vel_w, latent_w = params

    Data_collect = KoopmanDatasetCollector(env_name)
    udim, Nstate = Data_collect.u_dim, Data_collect.state_dim

    subsuffix  = "../../" + model_path[3:]
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

    observation_list, u_list = [], []
    observation = env.reset_state(reset_state)
    x0 = np.matrix(Psi_o(observation, net, NKoopman))
    x_ref_lift = Psi_o(x_ref, net, NKoopman)
    observation_list.append(np.array(x0[:Nstate]).reshape(-1, 1))

    failure = False
    for _ in range(steps):
        u = -Kopt @ (x0 - x_ref_lift)
        u_np = np.asarray(u).flatten()
        if action_clip is not None:
            u_np = np.clip(u_np, action_clip[0], action_clip[1])
        action = u_np[0] if u_np.size == 1 else u_np
        observation, _, _, _ = env.step(action)
        if Done(env_name, observation):
            failure = True; break
        x0 = np.matrix(Psi_o(observation, net, NKoopman))
        observation_list.append(np.array(x0[:Nstate]).reshape(-1, 1))
        u_list.append(u_np.astype(float)) 

    observations = np.concatenate(observation_list, axis=1)  # (Nstate, T)
    if failure:
        return None

    Q_state = np.array(Q_full)[:Nstate, :Nstate]
    R_eval  = uval * np.array(R)
    last10_L1, full_L1 = compute_tracking_metrics(observations, x_ref, angle_idx, last_N=lastN)
    J = cost_with_wrap(observations, u_list, Q_state, R_eval, x_ref, angle_idx)

    return {
        "last10_L1": last10_L1,
        "full_L1": full_L1,
        "control_cost": J,
        "params": {"uval": uval, "angle_w": angle_w, "vel_w": vel_w, "latent_w": latent_w}
    }

# ============================
# Updated grids (second grid)
# ============================
DAMPING_GRID = {
    "uval":     [1e-4, 1e-3, 1e-2, 1e-1, 1],
    "angle_w":  [1e-2, 1e-1, 1.0, 10.0],
    "vel_w":    [1e-1, 0, 1],
    "latent_w": [0.0],
}

DOUBLE_GRID = {
    "uval":     [1e-4, 1e-3, 1e-2, 1e-1, 1],
    "angle_w":  [1e-2, 1e-1, 1.0, 10.0],
    "vel_w":    [1e-1, 0, 1],
    "latent_w": [0.0],
}

def grid_for_env(env_name):
    return DOUBLE_GRID if env_name.startswith("DoublePendulum") else DAMPING_GRID

def ordered_uvals(env_name, ctrl_loss):
    g = grid_for_env(env_name)
    uvals = list(g["uval"])
    uvals.sort()  # ascending order (tiny first, which helps DP stability)
    return uvals

# ============================
# Single-stage grid search (no local refinement)
# ============================
def evaluate_model_grid(model_path, env_name, encode_dim, ctrl_loss,
                        steps=150, lastN=10, stability_rad=1.05,
                        action_clip=None, use_residual=True):
    grid = grid_for_env(env_name)
    uvals = ordered_uvals(env_name, ctrl_loss)

    best = None
    tried = 0

    # Single pass over (uval, angle_w, vel_w, latent_w)
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
                    if (best is None) or (out["last10_L1"] < best["last10_L1"]):
                        best = out

    return best, tried

# ============================
# Batch over models
# ============================
def main():
    project_name = "Aug_8"
    base_dir = f"../../log/{project_name}"
    os.makedirs(base_dir, exist_ok=True)

    log_path = f"{base_dir}/koopman_results_log.csv"
    log = pd.read_csv(log_path)
    log = log[(log['env_name'] == 'DampingPendulum') | (log['env_name'] == 'DoublePendulum')]

    rows = []
    for i in range(len(log)):
        row = log.iloc[i]
        model_path   = row['model_path']
        env_name     = row['env_name']
        encode_dim   = int(row['encode_dim'])
        ctrl_loss    = row.get('use_control_loss', None)
        cov_loss     = row.get('use_covariance_loss', None)
        seed         = row.get('seed', None)

        try:
            if env_name.startswith("DoublePendulum"):
                sr   = 1.1
                clip = None
                steps = 200
            else:
                sr   = 1.1
                clip = None
                steps = 100
            best, tried = evaluate_model_grid(
                model_path=model_path,
                env_name=env_name,
                encode_dim=encode_dim,
                ctrl_loss=ctrl_loss,
                steps=steps,
                lastN=10,
                stability_rad=sr,
                action_clip=clip,
                use_residual=True
            )

            total_tried = tried
            if best is None:
                print(f"❌ Failure: {env_name} | enc={encode_dim} | ctrl_loss={ctrl_loss} | cov_loss={cov_loss} | seed={seed} | (no stable candidate out of {total_tried})")
                rows.append({
                    'env_name': env_name, 'encode_dim': encode_dim,
                    'use_control_loss': ctrl_loss, 'use_covariance_loss': cov_loss, 'seed': seed,
                    'best_last10_L1': np.inf, 'best_full_L1': np.inf, 'best_control_cost': np.inf,
                    'best_uval': None, 'best_angle_w': None, 'best_vel_w': None, 'best_latent_w': None,
                    'tried_total': total_tried,
                    'model_path': model_path,
                    'stable': 0  # mark instability for aggregation
                })
            else:
                p = best["params"]
                print(
                    f"✅ Best: {env_name} | enc={encode_dim} | ctrl_loss={ctrl_loss} | cov_loss={cov_loss} | seed={seed} | "
                    f"last10_L1={best['last10_L1']:.4f} | full_L1={best['full_L1']:.4f} | J={best['control_cost']:.2f} | "
                    f"uval={p['uval']} angle_w={p['angle_w']} vel_w={p['vel_w']} latent_w={p['latent_w']} "
                    f"(from total {total_tried})"
                )
                rows.append({
                    'env_name': env_name, 'encode_dim': encode_dim,
                    'use_control_loss': ctrl_loss, 'use_covariance_loss': cov_loss, 'seed': seed,
                    'best_last10_L1': best['last10_L1'], 'best_full_L1': best['full_L1'], 'best_control_cost': best['control_cost'],
                    'best_uval': p['uval'], 'best_angle_w': p['angle_w'], 'best_vel_w': p['vel_w'], 'best_latent_w': p['latent_w'],
                    'tried_total': total_tried,
                    'model_path': model_path,
                    'stable': 1  # stable candidate found
                })

        except Exception as e:
            print(f"❌ Failed: {env_name} | enc={encode_dim} | ctrl_loss={ctrl_loss} | cov_loss={cov_loss} | seed={seed} | Error: {e}")
            rows.append({
                'env_name': env_name, 'encode_dim': encode_dim,
                'use_control_loss': ctrl_loss, 'use_covariance_loss': cov_loss, 'seed': seed,
                'best_last10_L1': np.inf, 'best_full_L1': np.inf, 'best_control_cost': np.inf,
                'best_uval': None, 'best_angle_w': None, 'best_vel_w': None, 'best_latent_w': None,
                'tried_stage1': 0, 'tried_stage2': 0, 'tried_total': 0,
                'model_path': model_path,
                'stable': 0
            })

    # -------------------------
    # Save per-model BEST results
    # -------------------------
    df_best = pd.DataFrame(rows)
    best_path = f"{base_dir}/pendulum_control_results_best.csv"
    df_best.to_csv(best_path, index=False)
    print(f"\nSaved BEST-per-model results to: {best_path}")

    # -------------------------
    # Build grouped summary with median & stability rate
    # Group by config keys (across seeds/models)
    # -------------------------
    group_keys = ['env_name', 'encode_dim', 'use_control_loss', 'use_covariance_loss']

    def _median_safe(x):
        # median over finite values; returns NaN if none are finite
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.median(x)) if x.size > 0 else np.nan

    def _mean_safe(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size > 0 else np.nan

    # counts
    grouped = df_best.groupby(group_keys, dropna=False)

    summary_rows = []
    for cfg, g in grouped:
        total = len(g)
        stable_cnt = int(g['stable'].sum())
        stability_rate = stable_cnt / total if total > 0 else np.nan

        med_last10 = _median_safe(g['best_last10_L1'])
        med_full   = _median_safe(g['best_full_L1'])
        med_J      = _median_safe(g['best_control_cost'])

        mean_last10 = _mean_safe(g['best_last10_L1'])
        mean_full   = _mean_safe(g['best_full_L1'])
        mean_J      = _mean_safe(g['best_control_cost'])

        # also helpful: how many had finite best_last10_L1
        finite_cnt = int(np.isfinite(g['best_last10_L1']).sum())

        summary = {
            'env_name': cfg[0],
            'encode_dim': cfg[1],
            'use_control_loss': cfg[2],
            'use_covariance_loss': cfg[3],
            'n_models': total,
            'n_stable': stable_cnt,
            'stability_rate': stability_rate,
            'finite_best_count': finite_cnt,
            'median_best_last10_L1': med_last10,
            'median_best_full_L1': med_full,
            'median_best_control_cost': med_J,
            'mean_best_last10_L1': mean_last10,
            'mean_best_full_L1': mean_full,
            'mean_best_control_cost': mean_J,
        }
        summary_rows.append(summary)

    df_summary = pd.DataFrame(summary_rows).sort_values(group_keys).reset_index(drop=True)

    # -------------------------
    # Save summary CSV with median & stability
    # -------------------------
    summary_path = f"{base_dir}/pendulum_control_results.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved grouped SUMMARY (with median & stability) to: {summary_path}")

if __name__ == "__main__":
    main()