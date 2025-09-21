# =========================
# Experiments to run (filters the LOG)
# =========================
project_name = "Aug_8"
encode_dims = [1, 2, 4, 8, 16]
cov_regs    = [0, 1]
ctrl_regs   = [0, 1]
train_samples = [1000, 4000, 16000, 60000]
seeds       = [17382, 76849, 20965, 84902, 51194]

import os
import sys
import numpy as np
import pandas as pd
import torch
import scipy.linalg
import pybullet as pb
from scipy.io import savemat

# local deps
sys.path.append("../utility")
from franka_env import FrankaEnv
from network import KoopmanNet
import lqr

# ========= Tunables (kept minimal) =========
NOISE_SIGMA       = 0.01
WARMUP_SKIP       = 100

# micro-selector
MICRO_STEPS       = 200
MICRO_WARMUP_SKIP = 25
R_GRID            = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 5e-1, 1.0, 2.0, 5.0]
TOP_K_FOR_MICRO   = 3

# gates
STABILITY_THRESH    = 1.10   # open-loop; relaxed
CLOSED_LOOP_MAX_RHO = 0.999  # warn if above, still proceed

# Q shaping (simple)
BASE_Q_FIRST_DIMS   = 10
LATENT_Q_WEIGHT     = 1e-3
LATENT_Q_WEIGHT_MID = 5e-3        # for encode_dim >= 68 or {4k,16k}
MID_SAMPLES_SET     = {4000, 16000}

# Feed-forward (regularized LS)
USE_FEEDFORWARD     = True
FEEDFORWARD_LAMBDA  = 1e-3
FEEDFORWARD_SCALE   = 1.0

# Clipping
USE_CLIP            = True

# ========= Small helpers =========
def spectral_radius(M: np.ndarray) -> float:
    return float(np.max(np.abs(scipy.linalg.eigvals(M))))

def build_Q(NKoop: int, in_dim: int, encode_dim: int, train_samples: int) -> np.ndarray:
    """Simple Q: identity on first dims + latent damping."""
    Q = np.zeros((NKoop, NKoop))
    Q[:min(BASE_Q_FIRST_DIMS, NKoop), :min(BASE_Q_FIRST_DIMS, NKoop)] = np.eye(min(BASE_Q_FIRST_DIMS, NKoop))
    latent_w = LATENT_Q_WEIGHT_MID if (encode_dim >= 68 or train_samples in MID_SAMPLES_SET) else LATENT_Q_WEIGHT
    Q[in_dim:, in_dim:] += latent_w * np.eye(encode_dim)
    return Q

def calc_feedforward_matrix(Bd: np.ndarray, lam: float) -> np.ndarray:
    """(Bd^T Bd + lam I)^-1 Bd^T  maps residual -> u_ff."""
    u_dim = Bd.shape[1]
    reg = lam * np.eye(u_dim)
    return np.linalg.solve(Bd.T @ Bd + reg, Bd.T)

def ik_to(env: FrankaEnv, x, y, z, threshold=1e-6, maxIter=10000):
    """Return first 7 joint angles by repeating IK until close."""
    it, close = 0, False
    x, y, z = float(x), float(y), float(z)
    while not close and it < maxIter:
        jointPoses = pb.calculateInverseKinematics(env.robot, env.ee_id, [x, y, z])
        for j in range(7):
            pb.resetJointState(env.robot, j, float(jointPoses[j]))
        newPos = pb.getLinkState(env.robot, env.ee_id)[4]
        dx, dy, dz = (x - newPos[0]), (y - newPos[1]), (z - newPos[2])
        close = (dx*dx + dy*dy + dz*dz) < threshold
        it += 1
    return jointPoses[:7]

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
    return sorted(list(set(out)))

# ========= Core evaluation =========
def calculate_control_tasks_err(encode_dim, use_residual, train_samples, model_path):
    # --- Load env/model ---
    env = FrankaEnv(render=False)
    in_dim, u_dim = env.Nstates, env.udim

    # load weights (handle PyTorch 2.6 weights_only change gracefully)
    full_path = "../../Koopman-Scaling/" + model_path[3:]
    try:
        payload = torch.load(full_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(full_path, map_location="cpu")
    state_dict, Elayer = payload["model"], payload["layer"]

    NKoop = encode_dim + in_dim
    net = KoopmanNet(Elayer, NKoop, u_dim, use_residual)
    net.load_state_dict(state_dict)
    net.cpu().double().eval()

    Ad = state_dict["lA.weight"].cpu().numpy()
    Bd = state_dict["lB.weight"].cpu().numpy()

    # --- Gates ---
    rhoA = spectral_radius(Ad)
    print(f"[INFO] rho(A)={rhoA:.6f}")
    if rhoA > STABILITY_THRESH:
        print("[WARN] Skip: rho(A) above threshold.")
        return rhoA, float("inf"), float("inf")

    # --- Time base & figure-8 refs ---
    T = 6 * 10
    t = 1.6 + 0.02 * np.linspace(0, T*5, T*50+1)
    Steps = len(t) - 1
    a = 0.3
    x = 0.3 * np.ones((len(t), 1))
    z = (0.59 + 2 * a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)).reshape(-1, 1)
    y = (a * np.cos(t) / (1 + np.sin(t)**2)).reshape(-1, 1)

    def Psi_enc(s):
        ds = net.encode(torch.as_tensor(s, dtype=torch.double)).detach().cpu().numpy().reshape(-1)
        psi = np.zeros((NKoop, 1)); psi[:NKoop, 0] = ds[:NKoop]
        return psi

    def Obs(o):
        noise = np.random.randn(3) * NOISE_SIGMA
        return np.concatenate((o[:3] + noise, o[7:]), axis=0), noise

    # IK for figure-8 desired states
    env.reset()
    J8 = np.empty((len(t), 7)); J8[:] = np.NaN
    for i in range(len(t)):
        J8[i, :] = ik_to(env, x[i], y[i], z[i])
    states_des = np.concatenate((x, y, z, J8, np.zeros((len(y), 7))), axis=1)  # (N, 17)

    # --- Q build ---
    Q = build_Q(encode_dim + in_dim, in_dim, encode_dim, train_samples)

    # --- Candidate Ks (R-grid) ---
    Ad_m, Bd_m = np.matrix(Ad), np.matrix(Bd)
    candidates = []
    for rscale in R_GRID:
        R = np.matrix(np.eye(u_dim) * rscale)
        try:
            Kc = lqr.lqr_regulator_k(Ad_m, Bd_m, np.matrix(Q), R)
        except Exception:
            continue
        rho_cl = spectral_radius(Ad - Bd @ np.asarray(Kc))
        candidates.append((rho_cl, rscale, Kc))
    if not candidates:
        print("[WARN] No K candidates found; marking inf.")
        return rhoA, float("inf"), float("inf")
    candidates.sort(key=lambda x: x[0])  # lowest closed-loop rho first
    top = candidates[:min(TOP_K_FOR_MICRO, len(candidates))]

    # --- Feed-forward operator ---
    FF = calc_feedforward_matrix(Bd, FEEDFORWARD_LAMBDA) if USE_FEEDFORWARD else None
    sat_val = getattr(env, "sat_val", 0.3)

    # --- Generic rollout (micro/full) ---
    def run_rollout(StepsRun, states_ref, warm_skip) -> (np.ndarray, float):
        state = env.reset()

        # small offsets
        a_off0, b_off0 = 0.01, 0.043
        y0 = float(states_ref[0, 1] + a_off0)
        z0 = float(states_ref[0, 2] + b_off0)

        # init joints
        j_init = ik_to(env, states_ref[0, 0], y0, z0)
        for j, jnt in enumerate(j_init):
            pb.resetJointState(env.robot, j, float(jnt))
        state = env.get_state()

        enc_traj = np.empty((StepsRun+1, encode_dim + in_dim)); enc_traj[:] = np.NaN

        for k in range(StepsRun):
            s_used, _ = Obs(state)
            enc = Psi_enc(s_used); enc_traj[k, :] = enc.reshape(-1)
            enc_ref_next = Psi_enc(states_ref[k+1, :])

            if FF is not None:
                resid = (enc_ref_next - Ad @ enc)     # NKoop x 1
                u_ff = FEEDFORWARD_SCALE * (FF @ resid).reshape(-1, 1)
            else:
                u_ff = 0.0

            du = u_ff - np.asarray(Kopt) @ (enc - enc_ref_next)
            if USE_CLIP: du = np.clip(du, -sat_val, +sat_val)
            state = env.step(du)

        s_used, _ = Obs(state); enc = Psi_enc(s_used); enc_traj[StepsRun, :] = enc.reshape(-1)

        yz_traj = enc_traj[:, [1, 2]]
        yz_ref  = states_ref[:, [1, 2]]
        if warm_skip > 0:
            yz_traj = yz_traj[warm_skip:]
            yz_ref  = yz_ref[warm_skip:]
        err = float(np.linalg.norm(yz_traj - yz_ref))
        return enc_traj, err

    # --- Micro selection ---
    def run_micro(K) -> float:
        # short rollout from first desired
        state = env.reset()
        for j, jnt in enumerate(states_des[0, 3:10]):
            pb.resetJointState(env.robot, j, float(jnt))
        state = env.get_state()

        steps = min(MICRO_STEPS, Steps)
        enc_traj = np.empty((steps+1, encode_dim + in_dim)); enc_traj[:] = np.NaN
        for k in range(steps):
            s_used, _ = Obs(state)
            enc = Psi_enc(s_used); enc_traj[k, :] = enc.reshape(-1)
            enc_ref_next = Psi_enc(states_des[k+1, :])

            if FF is not None:
                resid = (enc_ref_next - Ad @ enc)
                u_ff = FEEDFORWARD_SCALE * (FF @ resid).reshape(-1, 1)
            else:
                u_ff = 0.0

            du = u_ff - np.asarray(K) @ (enc - enc_ref_next)
            if USE_CLIP: du = np.clip(du, -sat_val, +sat_val)
            state = env.step(du)

        s_used, _ = Obs(state); enc = Psi_enc(s_used); enc_traj[steps, :] = enc.reshape(-1)
        yz_traj = enc_traj[:, [1, 2]]
        yz_ref  = states_des[:steps+1, [1, 2]]
        return float(np.linalg.norm(yz_traj[MICRO_WARMUP_SKIP:] - yz_ref[MICRO_WARMUP_SKIP:]))

    micro_scores = []
    for rho_cl, rscale, Kc in top:
        try:
            e_micro = run_micro(Kc)
        except Exception as e:
            print(f"[WARN] Micro rollout failed for R={rscale}: {e}")
            e_micro = float("inf")
        micro_scores.append((e_micro, rho_cl, rscale, Kc))

    micro_scores.sort(key=lambda x: (np.isinf(x[0]), x[0], x[1]))
    e_micro_best, rho_best, r_best, Kopt = micro_scores[0]
    print(f"[INFO] K selected: R={r_best}, rho(A-BK)={rho_best:.6f}, micro_err={e_micro_best:.3f}")

    if rho_best > CLOSED_LOOP_MAX_RHO:
        print(f"[WARN] Closed-loop rho={rho_best:.4f} > {CLOSED_LOOP_MAX_RHO:.4f} — proceeding anyway.")

    # --- Full Figure-8 rollout ---
    enc_traj, err_fig8 = run_rollout(Steps, states_des, WARMUP_SKIP)
    # os.makedirs("Results", exist_ok=True)
    # savemat("Results/DKUC_FrankaFig8noise_SimData0.mat", {
    #     "desired_states": states_des,
    #     "states_KoopmanU": enc_traj,
    #     "error_KoopmanU":  err_fig8
    # })
    print("[INFO] Fig-8 error =", err_fig8)

    # --- Star references & rollout ---
    center = np.array([0.0, 0.6]); radius = 0.3; theta_ = np.pi / 10.0
    eradius = np.tan(2*theta_) * radius * np.cos(theta_) - radius * np.sin(theta_)
    Star = np.zeros((11, 2))
    for i in range(5):
        th = 2*np.pi/5*(i+0.25); be = 2*np.pi/5*(i+0.75)
        Star[2*i]   = [np.cos(th)*radius + center[0], np.sin(th)*radius + center[1]]
        Star[2*i+1] = [np.cos(be)*eradius + center[0], np.sin(be)*eradius + center[1]]
    Star[-1] = Star[0]

    T2 = 6 * 10
    t2 = 0.02 * np.linspace(0, T2*5, T2*50+1)
    Steps2 = len(t2) - 1
    each_num = int((len(t2) - 10) / 9.5)
    refs = np.zeros((len(t2), 2))
    for i in range(10):
        refs[(each_num+1)*i] = Star[i]
        num = each_num if i != 9 else len(t2) - (each_num+1)*i - 1
        for j in range(num):
            tau = (j+1)/(each_num+1)
            refs[(each_num+1)*i + j + 1] = tau*Star[i+1] + (1-tau)*Star[i]

    x2 = 0.3 * np.ones((len(t2), 1))
    z2 = refs[:, 1].reshape(-1, 1)
    y2 = refs[:, 0].reshape(-1, 1)

    Jstar = np.empty((len(t2), 7)); Jstar[:] = np.NaN
    for i in range(len(t2)):
        Jstar[i, :] = ik_to(env, x2[i], y2[i], z2[i])
    states_des2 = np.concatenate((x2, y2, z2, Jstar, np.zeros((len(y2), 7))), axis=1)

    enc_traj2, err_star = run_rollout(Steps2, states_des2, WARMUP_SKIP)
    # savemat("Results/DKUC_FrankaFigStarnoise_SimData0.mat", {
    #     "desired_states": states_des2,
    #     "states_KoopmanU": enc_traj2,
    #     "error_KoopmanU":  err_star
    # })
    print("[INFO] Star error =", err_star)

    return rhoA, err_fig8, err_star

# ========= Main =========
if __name__ == "__main__":
    log_path = f"../../Koopman-Scaling/log/{project_name}/koopman_results_log.csv"
    out_path = f"../../Koopman-Scaling/log/{project_name}/franka_control_results.csv"

    # --- Load raw log ---
    raw = pd.read_csv(log_path)

    # --- Determine Franka input dim to map multipliers -> absolute sizes ---
    _tmp_env = FrankaEnv(render=False)
    in_dim = _tmp_env.Nstates  # expected 17
    del _tmp_env

    # map encode_dim multipliers to absolute latent sizes present in the log
    enc_abs = sorted({m * in_dim for m in encode_dims})

    # normalize boolean filters to True/False
    cov_bools  = to_bool_list(cov_regs)
    ctrl_bools = to_bool_list(ctrl_regs)

    # column checks
    need = {'env_name','encode_dim', 'train_samples', 'use_covariance_loss','use_control_loss','seed','model_path'}
    if not need.issubset(raw.columns):
        missing = sorted(need - set(raw.columns))
        raise KeyError(f"Log missing required columns: {missing}")

    # --- Apply filters ---
    log = raw[
        (raw["env_name"] == "Franka")
        & (raw["encode_dim"].isin(enc_abs))
        & (raw["train_samples"].isin(train_samples))
        & (raw["use_covariance_loss"].isin(cov_bools))
        & (raw["use_control_loss"].isin(ctrl_bools))
        & (raw["seed"].isin(seeds))
    ].reset_index(drop=True)

    if len(log) == 0:
        print("[Error] No models found after filtering.")
        def uniq(col):
            return sorted(raw[col].dropna().unique().tolist()) if col in raw.columns else "N/A"
        print("Available envs:", uniq("env_name"))
        print("Requested encode_dims (multipliers) -> absolute:", encode_dims, "->", enc_abs)
        print("Available encode_dim:", uniq("encode_dim"))
        print("Requested cov_regs ->", cov_regs, "normalized to", cov_bools, "; available:", uniq("use_covariance_loss"))
        print("Requested ctrl_regs ->", ctrl_regs, "normalized to", ctrl_bools, "; available:", uniq("use_control_loss"))
        print("Requested seeds:", seeds, "; available:", uniq("seed"))
        sys.exit(0)

    results = []
    for i in range(len(log)):
        row = log.iloc[i]
        encode_dim    = int(row["encode_dim"])
        use_residual  = bool(row["use_residual"]) if "use_residual" in row else True
        train_samples = int(row["train_samples"]) if "train_samples" in row and pd.notna(row["train_samples"]) else 0
        model_path    = row["model_path"]

        print("="*80)
        print(f"[MODEL] {model_path}")
        print(f"[INFO] encode_dim={encode_dim} (abs), in_dim={in_dim}, "
              f"train_samples={train_samples}, use_residual={use_residual}, "
              f"ctrl={row.get('use_control_loss','?')}, cov={row.get('use_covariance_loss','?')}, seed={row.get('seed','?')}")

        try:
            rhoA, e8, eS = calculate_control_tasks_err(encode_dim, use_residual, train_samples, model_path)
        except Exception as e:
            print(f"[ERROR] Exception during evaluation: {e}")
            rhoA, e8, eS = float("inf"), float("inf"), float("inf")

        results.append({
            "model_path":  model_path,
            "max_eig":     rhoA,
            "error_eight": e8,
            "error_star":  eS,
        })
        try:
            pb.disconnect()
        except Exception:
            pass

    results_df = pd.DataFrame(results)
    keep_cols = ["model_path","env_name","seed","train_samples","encode_dim",
                 "use_control_loss","use_covariance_loss"]
    results_merged = log[keep_cols].merge(results_df, on="model_path").drop(columns=["model_path"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results_merged.to_csv(out_path, index=False)
    print("Saved results to:", out_path)