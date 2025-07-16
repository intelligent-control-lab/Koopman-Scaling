'''Batch script to track reference trajectories using dkuc Koopman MPC on G1 and Go2 robots.'''
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

# Custom network code path (this is safe to import early)
sys.path.append('../utility')
from network import KoopmanNet

# -- Parser setup ------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Batch-track G1 and Go2 with dkuc Koopman MPC"
)

# IMPORTANT: Import AppLauncher before other Isaac imports
from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    '--csv_log_path', type=str,
    default='../log/Jun_11_control/koopman_results_log.csv',
    help='CSV file with columns [env_name, model_path, ...]'
)
parser.add_argument(
    '--save_path', type=str, default='../log/Jun_11_control/isaac_control_results.csv',
    help='Output CSV file for aggregated metrics'
)
parser.add_argument(
    '--start_idx', type=int, default=0,
    help='Start from this model index (useful for resuming)'
)
# Removed --horizon argument
args_cli = parser.parse_args()

# Launch Isaac Sim FIRST (this initializes the simulation)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# NOW import Isaac Lab modules (after simulation is initialized)
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.dict import print_dict
import gymnasium as gym

# Import the EventTermCfg for proper event configuration
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject

# Fallback wrapper if RSL-RL is unavailable
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

# Device selection
device = torch.device(
    'cuda:0' if torch.cuda.is_available() and not args_cli.cpu else 'cpu'
)

# Mapping from CSV env_name to Gym task
TASK_MAP = {
    'G1':  'Isaac-Velocity-Flat-G1-v0',
    'Go2': 'Isaac-Velocity-Flat-Unitree-Go2-v0',
}

# Reference trajectory filenames for each robot
REF_TRAJECTORIES = {
    'G1':  '2025-03-19_17-35-32_trajnum3000_trajlen500',
    'Go2': 'None_trajnum3000_trajlen500',
}

# Mapping to reference data paths
REF_PATH_FMT = {
    'G1':  '../data/g1_flat/reference_repository/{}.npz',
    'Go2': '../data/unitree_go2_flat/reference_repository/{}.npz',
}

# MPC horizon (trajectory length)
TRAJ_LEN = 15  # For G1 and Go2 robots

# -- Helper functions -------------------------------------------------------

def reset_root_state_specific(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset base pose/velocity from ref_data."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    global ref_data
    
    root_states = ref_data['state_data'][0, env_ids, -13:]
    
    if 'Rough' in env.__class__.__name__:
        positions = root_states[:, :3]
    else:
        positions = torch.cat([env.scene.env_origins[env_ids][:, :2], root_states[:, 2:3]], dim=-1)
    orientations = root_states[:, 3:7]
    velocities = root_states[:, 7:13]
    
    # Set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1).float(), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities.float(), env_ids=env_ids)


def reset_joints_specific(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joint positions/velocities from ref_data."""
    asset: Articulation = env.scene[asset_cfg.name]
    global ref_data
    
    joint_states = ref_data['state_data'][0, env_ids, :-13]
    
    joint_pos = joint_states[:, :joint_states.shape[1]//2]
    joint_vel = joint_states[:, joint_states.shape[1]//2:]
    
    # Set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos.float(), joint_vel.float(), env_ids=env_ids)


def compute_ref_dof_acc():
    """Compute DoF accelerations from ref velocities."""
    global ref_data, env_cfg, robot
    dt = env_cfg.sim.dt * env_cfg.decimation
    if 'g1' in robot:
        v = ref_data['state_data'][:, :, 37:60]
    else:
        v = ref_data['state_data'][:, :, 12:24]
    acc = torch.zeros_like(v).to(device)
    acc[1:] = (v[1:] - v[:-1]) / dt
    return acc


def Torch_MPC(net, x_k, X_ref, H, m, n):
    """Single-horizon Koopman MPC action computation."""
    z_k   = net.encode(x_k).permute(1,0)
    Z_ref = net.encode(X_ref).permute(1,0,2).reshape(-1, (H+1)*n)
    A = net.lA.weight.data; B = net.lB.weight.data
    Q = torch.eye(n, device=device, dtype=torch.double)
    R = torch.eye(m, device=device, dtype=torch.double) * 0.
    F = torch.eye(n, device=device, dtype=torch.double)
    # determine encoder input dim
    s_dim = getattr(net, 's_dim', x_k.shape[-1])
    Q[s_dim:,s_dim:]=0; F[s_dim:,s_dim:]=0
    # build block matrices
    M = torch.cat([torch.matrix_power(A,i) for i in range(H+1)],0).double()
    C = torch.zeros((H+1)*n, H*m, device=device, dtype=torch.double)
    for i in range(1,H+1):
        for j in range(i):
            C[i*n:(i+1)*n, j*m:(j+1)*m] = torch.matrix_power(A, i-j-1) @ B
    Qh = torch.block_diag(*([Q]*H + [F])); Rh = torch.block_diag(*([R]*H))
    P = 2*(Rh + C.T@Qh@C)
    q = 2*(z_k.T@M.T - Z_ref)@Qh@C
    U = (-0.5 * torch.inverse(0.5*P) @ q.T).reshape(H*m, -1)
    return U[:m].permute(1,0)


def load_koopman_model(path, state_dim_default, u_dim_default, device):
    """Load a Koopman model, inferring dimensions from checkpoint if possible."""
    chk = torch.load(path, map_location=device)
    
    # Try to infer dimensions from the checkpoint
    if 'model' in chk:
        model_dict = chk['model']
    else:
        model_dict = chk
        
    # Infer action dimension from lB layer if it exists
    if 'lB.weight' in model_dict:
        lB_shape = model_dict['lB.weight'].shape
        u_dim = lB_shape[1]  # Input dimension of lB is the action dimension
        print(f"[INFO] Inferred action dimension from checkpoint: {u_dim}")
    else:
        u_dim = u_dim_default
        print(f"[INFO] Using default action dimension: {u_dim}")
    
    # Try to get layer architecture from checkpoint
    layers = chk.get('layer', None)
    if layers is None:
        # Try to infer from lA layer shape
        if 'lA.weight' in model_dict:
            lA_shape = model_dict['lA.weight'].shape
            latent_dim = lA_shape[0]  # Output dimension of lA is the latent dimension
            enc_dim = latent_dim - state_dim_default  # Assuming latent = state + encoded
            print(f"[INFO] Inferred latent dimension: {latent_dim}, encode dimension: {enc_dim}")
        else:
            enc_dim = 64  # Default
            latent_dim = state_dim_default + enc_dim
            
        layers = [state_dim_default, 256, 256, 256, enc_dim]
    
    # Create model with inferred dimensions
    net = KoopmanNet(layers, layers[0] + layers[-1], u_dim,
                     use_residual=True).to(device).double()
    
    # Load state dict
    net.load_state_dict(model_dict)
    return net


def get_obs(env, normalize=False, mean=None, std=None):
    """Fetch and optionally normalize joint+root state."""
    base = getattr(env, 'unwrapped', getattr(env, 'env', env))
    d = base.scene['robot'].data
    s = torch.cat([d.joint_pos, d.joint_vel, d.root_state_w], dim=-1)
    if normalize and mean is not None and std is not None:
        s = ((s-mean)/std).squeeze(0)
    return s.double()


# -- Core tracking per-model ------------------------------------------------

def run_one_tracking(model_path, row):
    """
    Runs tracking for one model checkpoint. Returns metrics dict.
    """
    global ref_data, robot, env_cfg
    # determine robot env
    name = row['env_name']  # 'G1' or 'Go2'
    robot = name.lower()
    task  = TASK_MAP[name]
    
    # Get the appropriate reference trajectory for this robot
    ref_file = REF_TRAJECTORIES[name]
    
    # load reference
    ref_path = REF_PATH_FMT[name].format(ref_file)
    print(f"Loading reference trajectory from: {ref_path}")
    
    if not os.path.exists(ref_path):
        print(f"Error: Reference trajectory not found at {ref_path}")
        return None
        
    ref_np = np.load(ref_path, allow_pickle=True)
    state_data  = torch.DoubleTensor(ref_np['state_data']).to(device)
    action_data = torch.DoubleTensor(ref_np['action_data']).to(device)
    ref_data = OrderedDict({'state_data': state_data, 'action_data': action_data})
    
    # Get actual action dimension from reference data
    env_action_dim = action_data.shape[-1]  # This is what the environment expects
    print(f"[INFO] Environment expects {env_action_dim} action dimensions")
    
    # Determine state dimension based on robot type
    if 'g1' in robot:
        state_dim = 87  # G1 has 87 state dimensions
    else:  # go2
        state_dim = 37  # Go2 has 37 state dimensions  
        
    H = TRAJ_LEN  # Use hardcoded horizon
    
    # load model - it will infer the correct action dimension
    print(f"Loading Koopman model from: {model_path}")
    net = load_koopman_model(model_path, state_dim, env_action_dim, device)
    net.eval()
    
    # Get the actual action dimension used by the model
    model_action_dim = net.lB.weight.shape[1]
    print(f"[INFO] Model uses action dimension: {model_action_dim}")
    
    # Get the latent dimension (output of encoder)
    with torch.no_grad():
        test_input = torch.randn(1, state_dim).double().to(device)
        test_encoded = net.encode(test_input)
        latent_dim = test_encoded.shape[-1]
        print(f"[INFO] Latent dimension: {latent_dim}")
    
    # prepare env
    num_envs = min(state_data.shape[1], 1000)  # Limit environments to reduce memory usage
    env_cfg = parse_env_cfg(task, num_envs=num_envs)
    
    # Configure reset events with proper EventTerm
    env_cfg.events.reset_base = EventTerm(
        func=reset_root_state_specific,
        mode="reset",
        params={},
    )
    env_cfg.events.reset_robot_joints = EventTerm(
        func=reset_joints_specific,
        mode="reset",
        params={},
    )
    
    # Set episode length
    max_steps = state_data.shape[0] - H - 1
    env_cfg.episode_length_s = max_steps * env_cfg.sim.dt * env_cfg.decimation
    
    # Disable base contact termination
    env_cfg.terminations.base_contact = None
    
    # Create environment
    print("Creating environment...")
    env = gym.make(task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    # compute ref acc
    ref_acc = compute_ref_dof_acc()
    
    # initial obs & stats
    base = getattr(env, 'unwrapped', getattr(env, 'env', env))
    obs = get_obs(env)[:num_envs]  # Ensure we only use the number of envs we created
    last_vel = base.scene['robot'].data.joint_vel.clone()
    
    # tracking loop
    JrPE = []
    start = time.time()
    
    print(f"Running tracking for {max_steps} steps...")
    with torch.inference_mode():
        for i in tqdm(range(max_steps)):
            # slice ref window
            idxs = base.episode_length_buf[:num_envs] + torch.arange(H+1, device=device).unsqueeze(1)
            X_ref = state_data[idxs, torch.arange(num_envs)].double()
            
            # compute action - pass latent_dim instead of state_dim
            a = Torch_MPC(net, obs, X_ref, H, model_action_dim, latent_dim)
            
            # Handle dimension mismatch between model output and environment expectation
            if model_action_dim != env_action_dim:
                if model_action_dim < env_action_dim:
                    # Pad with zeros if model outputs fewer actions
                    padding = torch.zeros(a.shape[0], env_action_dim - model_action_dim, 
                                        dtype=a.dtype, device=a.device)
                    a = torch.cat([a, padding], dim=1)
                else:
                    # Truncate if model outputs more actions
                    a = a[:, :env_action_dim]
            
            # Clamp actions and convert to float
            action_l = action_data[:, :num_envs, :].min(dim=0).values.min(dim=0).values
            action_u = action_data[:, :num_envs, :].max(dim=0).values.max(dim=0).values
            a = a.clamp(action_l.double(), action_u.double()).float()
            
            # step
            _, _, dones, _ = env.step(a)
            obs = get_obs(env)[:num_envs]
            
            # metrics - compare only the first model_action_dim joints
            s = base.scene['robot'].data.joint_pos[:num_envs]
            compare_dim = min(model_action_dim, s.shape[1])
            JrPE.append(torch.abs(ref_data['state_data'][i, :num_envs, :compare_dim] - s[:, :compare_dim]).mean().item())
            
    end = time.time()
    
    # Properly close the environment
    env.close()
    
    # Force garbage collection to free GPU memory
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'model_path': model_path,
        'robot': name,
        'mean_JrPE': np.mean(JrPE),
        'time_s': end-start,
    }


# -- Main execution --------------------------------------------------------

if __name__ == '__main__':
    try:
        df = pd.read_csv(args_cli.csv_log_path)
        df = df[df.env_name.isin(['G1','Go2'])]
        
        if df.empty:
            print("No G1 or Go2 models found in the CSV file")
            sys.exit(1)
            
        print(f"Found {len(df)} models to evaluate")
        
        # Check if we're resuming from a partial run
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
                if res is not None:
                    results.append(res)
                    
                    # Save incrementally after each successful run
                    if os.path.exists(args_cli.save_path):
                        # Append to existing file
                        existing_df = pd.read_csv(args_cli.save_path)
                        new_df = pd.DataFrame([res])
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df.to_csv(args_cli.save_path, index=False)
                    else:
                        # Create new file
                        pd.DataFrame([res]).to_csv(args_cli.save_path, index=False)
                    
                    print(f"[INFO] Saved results incrementally to {args_cli.save_path}")
                else:
                    print(f"[ERROR] Failed to track model {row.model_path}")
                    
            except Exception as e:
                print(f"[ERROR] Exception while processing {row.model_path}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        print(f"\n[DONE] Processed {len(results)} new models")
        print(f"[DONE] All results saved to {args_cli.save_path}")
            
    except Exception as e:
        print(f"[ERROR] Script failed with exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()