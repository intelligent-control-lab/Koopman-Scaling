import argparse
import os
import sys
import time
from datetime import datetime
from collections import OrderedDict

import yaml
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Custom network code path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, os.path.join(_project_root, 'utility'))
from network import KoopmanNet
from mpc_core import MPCSolver, step_errors, METRICS

# -------------------- Argparse / Isaac App --------------------
parser = argparse.ArgumentParser(description="Batch-track G1 and Go2 with dkuc Koopman MPC")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
parser.add_argument('--csv_log_path', type=str,
    help='CSV with columns [env_name, model_path, ...]')
parser.add_argument('--save_path', type=str,
    help='Output CSV for aggregated metrics')
parser.add_argument('--start_idx', type=int, default=0, help='Start index (resume)')
parser.add_argument('--traj_start', type=int, default=0,
    help='First reference trajectory to track (the paper uses 0-999; use e.g. 1000 for a validation set)')
parser.add_argument('--num_trajs', type=int, default=1000, help='Number of reference trajectories to track')
parser.add_argument('--mpc_r', type=float, default=0.0,
    help='Input weight R = mpc_r * I in the MPC cost (the paper uses 0)')
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab AFTER sim init
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict
import gymnasium as gym
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject

try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    class RslRlVecEnvWrapper:
        def __init__(self, env):
            self.env = env
            self.num_envs = env.num_envs
            self.episode_length_buf = env.episode_length_buf
        def step(self, actions): return self.env.step(actions)
        def reset(self):          return self.env.reset()
        def close(self):          return self.env.close()

device = torch.device('cuda:0' if torch.cuda.is_available() and not args_cli.cpu else 'cpu')

# -------------------- Task / Refs --------------------
TASK_MAP = {
    'G1':  'Isaac-Velocity-Flat-G1-v0',
    'Go2': 'Isaac-Velocity-Flat-Unitree-Go2-v0',
}
REF_TRAJECTORIES = {
    'G1':  '2025-03-19_17-35-32_trajnum3000_trajlen500',
    'Go2': 'None_trajnum3000_trajlen500',
}
REF_PATH_FMT = {
    'G1':  '../data/g1_flat/reference_repository/{}.npz',
    'Go2': '../data/unitree_go2_flat/reference_repository/{}.npz',
}
TRAJ_LEN = 15  # MPC horizon

# -------------------- Trim helpers (must match dataset.py) --------------------
def trim_state_action(env_name, state, action):
    """Trim raw (untrimmed) state/action tensors along last dim.
    Shapes:
      state: (..., S_full), action: (..., U_full)
    Returns:
      state_t: (..., S_trim), action_t: (..., U_trim)
    """
    if env_name == "G1":
        # state: [q(0:23), dq(37:60), height(76), root_vel(81:87)] -> 23+23+1+6=53
        s = torch.cat([state[..., :23], state[..., 37:60], state[..., 76:77], state[..., 81:87]], dim=-1)
        a = action[..., :23]  # drop hand actions
        return s, a
    elif env_name == "Go2":
        # state: [q/dq(0:24), root(26:37)] -> 24 + 11 = 35
        s = torch.cat([state[..., :24], state[..., 26:37]], dim=-1)
        a = action  # all actions used
        return s, a
    else:
        return state, action

def trimmed_dims(env_name):
    if env_name == "G1":
        return 53, 23
    elif env_name == "Go2":
        return 35, 12
    else:
        return None, None

# -------------------- Normalization helpers --------------------
def to_torch_stats(stats, device, dtype=torch.double):
    """Stats may be numpy or list; convert to torch on device."""
    if stats is None:
        return None
    out = {}
    for k in ['state_mean','state_std','action_mean','action_std']:
        if k in stats and stats[k] is not None:
            arr = stats[k]
            if isinstance(arr, torch.Tensor):
                t = arr.to(device=device, dtype=dtype)
            else:
                t = torch.tensor(arr, device=device, dtype=dtype)
            out[k] = t
    return out

def normalize_state(x, norm_stats):
    if norm_stats is None: return x
    return (x - norm_stats['state_mean']) / norm_stats['state_std']

def denormalize_action(u_norm, norm_stats):
    if norm_stats is None: return u_norm
    return u_norm * norm_stats['action_std'] + norm_stats['action_mean']

# -------------------- Isaac reset fns (raw/untrimmed) --------------------
def reset_root_state_specific(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    global ref_data_raw
    root_states = ref_data_raw['state_data'][0, env_ids, -13:]
    if 'Rough' in env.__class__.__name__:
        positions = root_states[:, :3]
    else:
        positions = torch.cat([env.scene.env_origins[env_ids][:, :2], root_states[:, 2:3]], dim=-1)
    orientations = root_states[:, 3:7]
    velocities = root_states[:, 7:13]
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1).float(), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities.float(), env_ids=env_ids)

def reset_joints_specific(env: ManagerBasedEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    global ref_data_raw
    joint_states = ref_data_raw['state_data'][0, env_ids, :-13]
    joint_pos = joint_states[:, :joint_states.shape[1]//2]
    joint_vel = joint_states[:, joint_states.shape[1]//2:]
    asset.write_joint_state_to_sim(joint_pos.float(), joint_vel.float(), env_ids=env_ids)

# -------------------- Acceleration (raw indexing for metrics) --------------------
def compute_ref_dof_acc(env_name, ref_state, env_cfg):
    dt = env_cfg.sim.dt * env_cfg.decimation
    if env_name == 'G1':
        v = ref_state[:, :, 37:60]
    else:
        v = ref_state[:, :, 12:24]
    acc = torch.zeros_like(v).to(device)
    acc[1:] = (v[1:] - v[:-1]) / dt
    return acc

def compute_dof_acc(dof_vel, last_vel, env_cfg):
    dt = env_cfg.sim.dt * env_cfg.decimation
    return (dof_vel - last_vel) / dt

# -------------------- MPC --------------------
def Torch_MPC(net, x_k_norm, X_ref_norm, H, m, latent_dim, s_dim):
    """Koopman MPC with normalized states/controls.
    Inputs are already normalized and trimmed.
    """
    z_k   = net.encode(x_k_norm).permute(1,0)                       # (batch, latent) -> (latent, batch)
    Z_ref = net.encode(X_ref_norm).permute(1,0,2).reshape(-1, (H+1)*latent_dim)

    A = net.lA.weight.data
    B = net.lB.weight.data

    # Costs in latent space; penalize only the first s_dim (original state part)
    Q = torch.eye(latent_dim, device=device, dtype=torch.double)
    R = torch.eye(m, device=device, dtype=torch.double) * args_cli.mpc_r
    F = torch.eye(latent_dim, device=device, dtype=torch.double)
    Q[s_dim:, s_dim:] = 0.0
    F[s_dim:, s_dim:] = 0.0

    # Block matrices
    M = torch.cat([torch.matrix_power(A, i) for i in range(H+1)], 0).double()
    C = torch.zeros((H+1)*latent_dim, H*m, device=device, dtype=torch.double)
    for i in range(1, H+1):
        for j in range(i):
            C[i*latent_dim:(i+1)*latent_dim, j*m:(j+1)*m] = torch.matrix_power(A, i-j-1) @ B

    Qh = torch.block_diag(*([Q]*H + [F]))
    Rh = torch.block_diag(*([R]*H))
    P = 2*(Rh + C.T @ Qh @ C)
    q = 2*(z_k.T @ M.T - Z_ref) @ Qh @ C
    U = (-0.5 * torch.inverse(0.5*P) @ q.T).reshape(H*m, -1)
    return U[:m].permute(1,0)  # (batch, m)

# -------------------- Checkpoint loading --------------------
def load_koopman_model_and_stats(path, env_name, device):
    chk = torch.load(path, weights_only=False, map_location=device)
    model_dict = chk['model'] if 'model' in chk else chk
    # Action dim from lB
    if 'lB.weight' in model_dict:
        u_dim = model_dict['lB.weight'].shape[1]
    else:
        u_dim = trimmed_dims(env_name)[1]
    # Layers
    layers = chk.get('layer', None)
    if layers is None:
        # infer encoder size from A
        if 'lA.weight' in model_dict:
            latent_dim = model_dict['lA.weight'].shape[0]
            # We don't know s_dim; use trimmed s_dim by env_name
            s_dim = trimmed_dims(env_name)[0]
            enc_dim = max(latent_dim - s_dim, 1)
            layers = [s_dim, 256, 256, 256, enc_dim]
        else:
            s_dim = trimmed_dims(env_name)[0]
            layers = [s_dim, 256, 256, 256, 64]
    net = KoopmanNet(layers, layers[0] + layers[-1], u_dim, use_residual=True).to(device).double()
    net.load_state_dict(model_dict)

    # Norm stats (trimmed dims)
    norm_stats = None
    if 'norm_stats' in chk:
        ns = chk['norm_stats']
        # Expect keys: state_mean, state_std, action_mean, action_std (already trimmed)
        norm_stats = to_torch_stats(ns, device, dtype=torch.double)

    return net, norm_stats

# -------------------- Observation helpers --------------------
def get_raw_obs_vec(env):
    base = getattr(env, 'unwrapped', getattr(env, 'env', env))
    d = base.scene['robot'].data
    # raw (untrimmed) layout: [joint_pos, joint_vel, root_state_w]
    return torch.cat([d.joint_pos, d.joint_vel, d.root_state_w], dim=-1)

# -------------------- Main tracking per model --------------------
def run_one_tracking(model_path, row):
    global ref_data_raw
    name = row['env_name']  # 'G1' or 'Go2'
    task = TASK_MAP[name]

    # Load reference repository (RAW, untrimmed)
    ref_path = REF_PATH_FMT[name].format(REF_TRAJECTORIES[name])
    if not os.path.exists(ref_path):
        print(f"[ERROR] Missing reference: {ref_path}")
        return None
    ref_np = np.load(ref_path, allow_pickle=True)
    sl = slice(args_cli.traj_start, args_cli.traj_start + args_cli.num_trajs)
    state_data_raw  = torch.as_tensor(ref_np['state_data'][:, sl], dtype=torch.double, device=device)
    action_data_raw = torch.as_tensor(ref_np['action_data'][:, sl], dtype=torch.double, device=device)
    ref_data_raw = OrderedDict(state_data=state_data_raw, action_data=action_data_raw)

    # Trimmed reference for CONTROL path
    state_t_ctrl, action_t_ctrl = trim_state_action(name, state_data_raw, action_data_raw)
    S_trim, U_trim = trimmed_dims(name)
    assert state_t_ctrl.shape[-1] == S_trim and action_t_ctrl.shape[-1] == U_trim

    # Environment action dimension (raw, untrimmed)
    env_action_dim = action_data_raw.shape[-1]

    # Load model + norm stats
    net, norm_stats = load_koopman_model_and_stats(model_path, name, device)
    net.eval()

    # Derived sizes
    with torch.no_grad():
        dummy = torch.zeros(1, S_trim, dtype=torch.double, device=device)
        latent_dim = net.encode(dummy).shape[-1]
    s_dim = S_trim
    m_dim = net.lB.weight.shape[1]  # trimmed action dim used by the model

    # Build env
    num_envs = state_data_raw.shape[1]
    env_cfg = parse_env_cfg(task, num_envs=num_envs)
    env_cfg.events.reset_base = EventTerm(func=reset_root_state_specific, mode="reset", params={})
    env_cfg.events.reset_robot_joints = EventTerm(func=reset_joints_specific, mode="reset", params={})
    max_steps = state_data_raw.shape[0] - TRAJ_LEN - 1
    env_cfg.episode_length_s = max_steps * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.terminations.base_contact = None
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Ref accelerations (raw) for JrAE metric
    ref_acc_raw = compute_ref_dof_acc(name, state_data_raw, env_cfg)

    # Initial obs (raw for metrics) & last_vel for acceleration
    base = getattr(env, 'unwrapped', getattr(env, 'env', env))
    obs_raw = get_raw_obs_vec(env)[:num_envs]  # (N, S_full)
    last_vel = base.scene['robot'].data.joint_vel.clone()

    # Metric buffers
    JrPE_list = [[] for _ in range(num_envs)]
    JrVE_list = [[] for _ in range(num_envs)]
    JrAE_list = [[] for _ in range(num_envs)]
    RPE_list  = [[] for _ in range(num_envs)]
    ROE_list  = [[] for _ in range(num_envs)]
    RLVE_list = [[] for _ in range(num_envs)]
    RAVE_list = [[] for _ in range(num_envs)]
    survival_steps = [0.0 for _ in range(num_envs)]
    collect_done   = [False for _ in range(num_envs)]

    threshold = 0.16 if name == "Go2" else 0.10
    start = time.time()

    print(f"[INFO] Tracking {name} | normalized={norm_stats is not None} | steps={max_steps} | num_envs={num_envs}")
    solver = MPCSolver(net, TRAJ_LEN, m_dim, latent_dim, s_dim, R_scale=args_cli.mpc_r, device=device)
    step_err = {k: [] for k in METRICS}                       # per step: (N,) tensors
    surv_t = torch.zeros(num_envs, dtype=torch.double, device=device)
    done_t = torch.zeros(num_envs, dtype=torch.bool, device=device)
    with torch.inference_mode():
        for i in tqdm(range(max_steps)):
            # ---------- Raw signals for metrics (all environments at once; same values as the per-env loop) ----------
            dof_pos = base.scene['robot'].data.joint_pos.clone()
            dof_vel = base.scene['robot'].data.joint_vel.clone()
            root_st = base.scene['robot'].data.root_state_w.clone()
            state_full = torch.cat([dof_pos, dof_vel, root_st], dim=1)
            if i != 0 and hasattr(env, 'dones'):
                d = env.dones[:num_envs].bool()
                last_vel[d] = dof_vel[d]
            e = step_errors(name, state_data_raw[i, :num_envs], ref_acc_raw[i, :num_envs],
                            state_full[:num_envs], compute_dof_acc(dof_vel, last_vel, env_cfg)[:num_envs])
            for k in METRICS:
                step_err[k].append(e[k])
            done_t |= e["JrPE"] > threshold
            surv_t += (~done_t).double()

            # ---------- Trim + Normalize for CONTROL ----------
            # ref window indices
            idxs = base.episode_length_buf[:num_envs] + torch.arange(TRAJ_LEN+1, device=device).unsqueeze(1)
            X_ref_raw = state_data_raw[idxs, torch.arange(num_envs)].double()  # (H+1, N, S_full)
            # trim
            X_ref_t, _ = trim_state_action(name, X_ref_raw, action_data_raw[:1, :num_envs])  # (H+1, N, S_trim)
            # normalize
            X_ref_norm = normalize_state(X_ref_t, norm_stats) if norm_stats is not None else X_ref_t

            # current obs: raw -> trim -> normalize
            obs_raw = get_raw_obs_vec(env)[:num_envs]
            obs_t, _ = trim_state_action(name, obs_raw, action_data_raw[:1, :num_envs])
            obs_norm = normalize_state(obs_t, norm_stats) if norm_stats is not None else obs_t

            # ---------- MPC in normalized space ----------
            a_norm = solver(obs_norm, X_ref_norm)  # (N, m_dim); identical to Torch_MPC, matrices cached

            # Denormalize to env units, then match env action dim
            a_env_trim = denormalize_action(a_norm, norm_stats) if norm_stats is not None else a_norm  # (N, U_trim)

            # Pad/truncate to env action dimension (raw)
            if m_dim < env_action_dim:
                pad = torch.zeros(a_env_trim.shape[0], env_action_dim - m_dim, dtype=a_env_trim.dtype, device=a_env_trim.device)
                a_env = torch.cat([a_env_trim, pad], dim=1)
            else:
                a_env = a_env_trim[:, :env_action_dim]

            # Clamp to ref action range (raw units)
            action_l = action_data_raw[:, :num_envs, :].min(dim=0).values.min(dim=0).values
            action_u = action_data_raw[:, :num_envs, :].max(dim=0).values.max(dim=0).values
            a_env = a_env.clamp(action_l.double(), action_u.double()).float()

            # Step
            _, _, dones, _ = env.step(a_env)
            env.dones = dones
            last_vel = dof_vel.clone()

    end = time.time()

    # Averages over time steps, per environment
    per_env = {k: torch.stack(step_err[k]).mean(dim=0).cpu().numpy() for k in METRICS}
    JrPE_list, JrVE_list, JrAE_list = per_env["JrPE"], per_env["JrVE"], per_env["JrAE"]
    RPE_list, ROE_list, RLVE_list, RAVE_list = per_env["RPE"], per_env["ROE"], per_env["RLVE"], per_env["RAVE"]
    survival_steps = surv_t.cpu().numpy()

    env.close()
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {
        'model_path': model_path,
        'robot': name,
        'traj_start': args_cli.traj_start,
        'num_trajs': num_envs,
        'mpc_r': args_cli.mpc_r,
        'mean_JrPE': np.mean(JrPE_list),
        'mean_JrVE': np.mean(JrVE_list),
        'mean_JrAE': np.mean(JrAE_list),
        'mean_RPE': np.mean(RPE_list),
        'mean_ROE': np.mean(ROE_list),
        'mean_RLVE': np.mean(RLVE_list),
        'mean_RAVE': np.mean(RAVE_list),
        'mean_survival_steps': np.mean(survival_steps),
        'time_s': end - start,
    }

# -------------------- Main --------------------
if __name__ == '__main__':
    try:
        df = pd.read_csv(args_cli.csv_log_path)
        df = df[df.env_name.isin(['G1','Go2'])]
        if df.empty:
            print("No G1 or Go2 models found in the CSV file")
            sys.exit(1)

        print(f"Found {len(df)} models to evaluate")

        if os.path.exists(args_cli.save_path):
            existing_df = pd.read_csv(args_cli.save_path)
            print(f"Found existing results with {len(existing_df)} models already processed")
            processed_paths = set(existing_df['model_path'].values)
        else:
            processed_paths = set()

        results = []
        for idx, row in df.iterrows():
            if idx < args_cli.start_idx:
                continue
            if row['model_path'] in processed_paths:
                print(f"\n[INFO] Skipping already processed: {row.env_name} -> {row.model_path}")
                continue

            print(f"\n[INFO] Processing model {idx+1}/{len(df)}: {row.env_name} -> {row.model_path}")
            try:
                res = run_one_tracking(row['model_path'], row)
                if res is None:
                    print(f"[ERROR] Failed to track model {row.model_path}")
                    continue

                results.append(res)
                # incremental save
                if os.path.exists(args_cli.save_path):
                    existing_df = pd.read_csv(args_cli.save_path)
                    new_df = pd.DataFrame([res])
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(args_cli.save_path, index=False)
                else:
                    pd.DataFrame([res]).to_csv(args_cli.save_path, index=False)

                print(f"[INFO] Saved results incrementally to {args_cli.save_path}")
                print(f"[INFO] Metrics: JrPE={res['mean_JrPE']:.4f}, JrVE={res['mean_JrVE']:.4f}, "
                      f"JrAE={res['mean_JrAE']:.4f}, survival_steps={res['mean_survival_steps']:.1f}")

            except Exception as e:
                print(f"[ERROR] Exception while processing {row['model_path']}: {e}")
                import traceback; traceback.print_exc()
                continue

        print(f"\n[DONE] Processed {len(results)} new models")
        print(f"[DONE] All results saved to {args_cli.save_path}")

    except Exception as e:
        print(f"[ERROR] Script failed with exception: {e}")
        import traceback; traceback.print_exc()
    finally:
        simulation_app.close()