"""Script to track reference trajectories using Koopman models for G1 and Go2 robots."""

import argparse
import os
import sys
import torch
import numpy as np
import yaml
import pandas as pd
import csv
from datetime import datetime
import time
from tqdm import tqdm
from collections import OrderedDict

# Add necessary paths to import custom modules
sys.path.append('../utility')
from network import KoopmanNet
from network import ResidualBlock  # Import for model loading

# MIGRATED: Updated Isaac Sim 4.5 imports
from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Track reference trajectories using Koopman models.")

# Add launcher arguments first (this will add --cpu and other Isaac Sim arguments)
AppLauncher.add_app_launcher_args(parser)

# Add your custom arguments (removed --cpu since it's already added by AppLauncher)
parser.add_argument("--video", action="store_true", default=False, help="Record videos during tracking.")
parser.add_argument("--video_length", type=int, default=999, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=1, help="Interval between video recordings (in steps).")
parser.add_argument("--task", type=str, help="Name of the task.")
parser.add_argument("--seed", type=int, default=0, help="Seed used for the environment")
parser.add_argument("--num_steps_per_env", type=int, default=200, help="RL Policy update interval")
parser.add_argument("--algo", type=str, choices=['incremental', 'dkuc', 'dkac', 'gmm', 'nndm'], help="Algorithm type")
parser.add_argument("--log_path", type=str, default="final_logs")
parser.add_argument("--save_path", type=str, default="final_metrics")
parser.add_argument("--ref", type=str, default="trajnum500")
parser.add_argument("--viewer", nargs='+', type=float, default=[])
parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to load")
parser.add_argument("--load_run", type=str, help="Specific run to load")
parser.add_argument("--model_path", type=str, help="Direct path to model file")
parser.add_argument("--test_mode", action="store_true", default=False, help="Run in test mode with dummy data")
parser.add_argument("--csv_log_path", type=str, default="../log/May_7_control_loss/koopman_results_log.csv", 
                    help="Path to CSV log file containing model paths")
args_cli = parser.parse_args()

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Launch omniverse app with headless mode if no display
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch.nn as nn

# MIGRATED: Updated Isaac Sim 4.5 imports
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import parse_env_cfg

# MIGRATED: Correct import path for Isaac Lab 4.5 RSL-RL wrapper
try:
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlVecEnvWrapper,
    )
    print("[INFO] Successfully imported RSL-RL wrapper from isaaclab_rl.rsl_rl")
except ImportError as e:
    print(f"[ERROR] Failed to import RSL-RL wrapper: {e}")
    
    # Create a basic wrapper as fallback
    class RslRlVecEnvWrapper:
        """Basic wrapper for RSL-RL compatibility"""
        def __init__(self, env):
            self.env = env
            self.num_envs = env.num_envs
            self.episode_length_buf = env.episode_length_buf
        
        def step(self, actions):
            return self.env.step(actions)
        
        def reset(self):
            return self.env.reset()
        
        def close(self):
            return self.env.close()
    
    class RslRlOnPolicyRunnerCfg:
        """Placeholder for RSL-RL runner config"""
        pass
    
    print("[INFO] Using basic fallback wrapper implementation")
# MIGRATED: Updated class names - Articulation is now the tensorized version, SingleArticulation for single objects
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ViewerCfg

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() and not args_cli.cpu else "cpu")

# Global variables
ref_data = {}
x_ref = None
robot = ""
ref_dof_acc = None  # Adding this missing global variable

def reset_root_state_specific(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a specific position and velocity from ref_data."""
    # MIGRATED: RigidObject and Articulation are now the tensorized versions
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    global ref_data

    root_states = ref_data["state_data"][0, env_ids, -13:]

    if "Rough" in args_cli.task:
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
    """Reset the robot joints using ref_data."""
    asset: Articulation = env.scene[asset_cfg.name]
    
    global ref_data

    joint_states = ref_data["state_data"][0, env_ids, :-13]

    joint_pos = joint_states[:, :joint_states.shape[1]//2]
    joint_vel = joint_states[:, joint_states.shape[1]//2:]

    # Set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos.float(), joint_vel.float(), env_ids=env_ids)


def compute_ref_dof_acc():
    """Compute reference DoF accelerations from velocity differences."""
    global ref_data
    
    dt = env_cfg.sim.dt * env_cfg.decimation
    if "g1" in robot:
        ref_dof_vel = ref_data["state_data"][:, :, 37:60]
    elif "go2" in robot or "a1" in robot or "anymal-D" in robot:
        ref_dof_vel = ref_data["state_data"][:, :, 12:24]
    else:
        ref_dof_vel = ref_data["state_data"][:, :, 12:24]  # Default case
        
    ref_dof_acc = torch.zeros_like(ref_dof_vel).to(device)
    ref_dof_acc[1:, ...] = (ref_dof_vel[1:, ...] - ref_dof_vel[:-1, ...]) / dt
    return ref_dof_acc


def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


def Torch_MPC(net, x_k, X_ref, H, m, n, device, robot):
    """MPC solver using Torch operations for efficient computation."""
    z_k = net.encode(x_k).permute(1, 0)
    Z_ref = net.encode(X_ref).permute(1, 0, 2).reshape(-1, (H+1)*n)

    A = net.lA.weight.data
    B = net.lB.weight.data
    Q = torch.eye(n).to(device).double()
    R = torch.eye(m).to(device).double() * 0.0
    F = torch.eye(n).to(device).double()

    # Get the state dimension from the network architecture
    # The s_dim should be the input dimension to the encoder
    if hasattr(net, 's_dim'):
        s_dim = net.s_dim
    elif hasattr(net, 'state_dim'):
        s_dim = net.state_dim
    else:
        # Infer from the network architecture - typically the first layer input size
        s_dim = A.shape[0] - (A.shape[0] - x_k.shape[-1])  # Rough estimate
        if s_dim <= 0 or s_dim > n:
            s_dim = x_k.shape[-1]  # Use input dimension as fallback
    
    Q[s_dim:, s_dim:] *= 0.0
    F[s_dim:, s_dim:] *= 0.0

    M = torch.cat([torch.matrix_power(A, i) for i in range(H+1)], dim=0).to(device).double()
    C = torch.zeros((H+1)*n, H*m)
    for i in range(1, H+1):
        for j in range(H):
            if j <= i - 1:
                C[i*n:(i+1)*n, j*m:(j+1)*m] = torch.matrix_power(A, i-j-1) @ B
    C = C.to(device).double()

    Q_hat = torch.block_diag(*([Q]*H + [F]))
    R_hat = torch.block_diag(*([R]*H))

    p = 2 * (R_hat + C.T @ Q_hat @ C)
    q = 2 * (z_k.T @ M.T - Z_ref) @ Q_hat @ C

    U_k = (-0.5 * torch.inverse(0.5*p) @ q.T).reshape(H*m, -1)
    u_k = U_k[:m, :].permute(1, 0)

    return u_k
    
def Torch_MPC_DKAC(net, x_k, X_ref, H, m, n, device, robot):
    """MPC solver for DKAC algorithm."""
    z_k = net.encode(x_k).permute(1, 0)
    Z_ref = net.encode(X_ref).permute(1, 0, 2).reshape(-1, (H+1)*n)

    A = net.lA.weight.data
    B = net.lB.weight.data
    Q = torch.eye(n).to(device).double()
    R = torch.eye(m).to(device).double() * 0.0
    F = torch.eye(n).to(device).double()

    # Get the state dimension from the network architecture
    if hasattr(net, 's_dim'):
        s_dim = net.s_dim
    elif hasattr(net, 'state_dim'):
        s_dim = net.state_dim
    else:
        s_dim = x_k.shape[-1]  # Use input dimension as fallback

    Q[s_dim:, s_dim:] *= 0.0
    F[s_dim:, s_dim:] *= 0.0

    M = torch.cat([torch.matrix_power(A, i) for i in range(H+1)], dim=0).to(device).double()
    C = torch.zeros((H+1)*n, H*m)
    for i in range(1, H+1):
        for j in range(H):
            if j <= i - 1:
                C[i*n:(i+1)*n, j*m:(j+1)*m] = torch.matrix_power(A, i-j-1) @ B
    C = C.to(device).double()

    Q_hat = torch.block_diag(*([Q]*H + [F]))
    R_hat = torch.block_diag(*([R]*H))

    p = 2 * (R_hat + C.T @ Q_hat @ C)
    q = 2 * (z_k.T @ M.T - Z_ref) @ Q_hat @ C

    U_k = (-0.5 * torch.inverse(0.5*p) @ q.T).reshape(H*m, -1)
    u_k = U_k[:m, :].permute(1, 0)

    z_k = z_k.permute(1, 0)
    g_x = net.action_encode(z_k[:, :s_dim])

    return u_k / g_x

def Torch_MPC_GMM(net, x_k, X_ref, H, m, n, device, robot, gmm_idx):
    """MPC solver for GMM algorithm."""
    z_k = net.encode(x_k).permute(1, 0)
    Z_ref = net.encode(X_ref).permute(1, 0, 2).reshape(-1, (H+1)*n)

    A = getattr(net, f'lA_{gmm_idx}').weight.data
    B = getattr(net, f'lB_{gmm_idx}').weight.data
    Q = torch.eye(n).to(device).double()
    R = torch.eye(m).to(device).double() * 0.0
    F = torch.eye(n).to(device).double()

    # Get the state dimension from the network architecture
    if hasattr(net, 's_dim'):
        s_dim = net.s_dim
    elif hasattr(net, 'state_dim'):
        s_dim = net.state_dim
    else:
        s_dim = x_k.shape[-1]  # Use input dimension as fallback

    Q[s_dim:, s_dim:] *= 0.0
    F[s_dim:, s_dim:] *= 0.0

    M = torch.cat([torch.matrix_power(A, i) for i in range(H+1)], dim=0).to(device).double()
    C = torch.zeros((H+1)*n, H*m)
    for i in range(1, H+1):
        for j in range(H):
            if j <= i - 1:
                C[i*n:(i+1)*n, j*m:(j+1)*m] = torch.matrix_power(A, i-j-1) @ B
    C = C.to(device).double()

    Q_hat = torch.block_diag(*([Q]*H + [F]))
    R_hat = torch.block_diag(*([R]*H))

    p = 2 * (R_hat + C.T @ Q_hat @ C)
    q = 2 * (z_k.T @ M.T - Z_ref) @ Q_hat @ C

    U_k = (-0.5 * torch.inverse(0.5*p) @ q.T).reshape(H*m, -1)
    u_k = U_k[:m, :].permute(1, 0)

    res = (M@z_k+C@U_k-Z_ref.T).T@Q_hat@(M@z_k+C@U_k-Z_ref.T)
    res = res[torch.arange(res.shape[0]), torch.arange(res.shape[0])]

    return u_k, res

def NMPC(net, x_k, X_ref, H, m, n, device, robot, u_min, u_max):
    """NMPC solver for NNDM algorithm."""
    K = 500

    Q = torch.eye(n).to(device).double()
    R = torch.eye(m).to(device).double() * 0.0
    F = torch.eye(n).to(device).double()
    Q_hat = torch.block_diag(*([Q]*H + [F]))
    R_hat = torch.block_diag(*([R]*H))

    def system_dynamics(x, u):
        return net(x, u)
    
    best_res = torch.ones(X_ref.shape[1]).to(device).double() * 1e9
    best_U_k = torch.zeros(H, X_ref.shape[1], m).to(device).double()
    for _ in range(K):
        X_k = torch.zeros_like(X_ref).to(device)
        X_k[0, ...] = x_k
        U_k = torch.rand((H, X_k.shape[1], m)).to(device) * (u_max - u_min) + u_min
        for j in range(H):
            X_k[j+1, ...] = system_dynamics(X_k[j], U_k[j])
        X_diff = (X_k - X_ref).permute(0, 2, 1).reshape((H+1)*n, -1)
        res = X_diff.T @ Q_hat @ X_diff
        res = res[torch.arange(res.shape[0]), torch.arange(res.shape[0])]
        idx = res < best_res
        best_res[idx] = res[idx]
        best_U_k[:, idx, :] = U_k[:, idx, :]
    best_U_k = best_U_k.permute(0, 2, 1).reshape(H*m, -1)
    u_k = best_U_k[:m, :].permute(1, 0)
    return u_k

def load_koopman_model(model_path, state_dim, u_dim, device):
    """Load a Koopman model from the given path."""
    # Extract model details from filename
    checkpoint = torch.load(model_path, map_location=device)
    layers = checkpoint.get('layer', None)
    
    # If layers information is not in the checkpoint, use the default architecture
    if layers is None:
        # Determine parameters from model path or use defaults
        encode_dim = 64
        hidden_layers = 3
        hidden_dim = 256
        # Create model with default architecture
        net = KoopmanNet([state_dim, hidden_dim, hidden_dim, hidden_dim, encode_dim], 
                        state_dim + encode_dim, u_dim, use_residual=True).to(device).double()
    else:
        # Create model with architecture from checkpoint
        net = KoopmanNet(layers, layers[0] + layers[-1], u_dim, use_residual=True).to(device).double()
    
    # Load state dict
    if 'model' in checkpoint:
        net.load_state_dict(checkpoint['model'])
    else:
        net.load_state_dict(checkpoint)
        
    return net

def find_model_paths_from_csv(csv_path, env_name):
    """Find model paths for the given environment from the CSV log file."""
    try:
        df = pd.read_csv(csv_path)
        # Filter for the specified environment
        env_df = df[df['env_name'] == env_name]
        
        if env_df.empty:
            print(f"No models found for environment {env_name}")
            return []
        
        # Sort by best validation loss
        env_df = env_df.sort_values(by='best_val_Kloss')
        
        # Return the full dataframe row for the best model (contains all parameters)
        return env_df.iloc[0]  # Return the best model's row as a Series
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def get_obs(env, normalize=False, state_mean=None, state_std=None):
    """Get normalized observation from the environment."""
    # MIGRATED: Updated environment access for Isaac Lab 4.5
    if hasattr(env, 'unwrapped'):
        # If wrapped environment, get the unwrapped version
        base_env = env.unwrapped
    elif hasattr(env, 'env'):
        # If it's a wrapper with .env attribute
        base_env = env.env
    else:
        # Direct environment
        base_env = env
        
    # Access the scene data
    robot_data = base_env.scene['robot'].data
    dof_pos = robot_data.joint_pos.clone()
    dof_vel = robot_data.joint_vel.clone()
    root_state = robot_data.root_state_w.clone()
    states = torch.cat([dof_pos, dof_vel, root_state], axis=-1)
    
    if "g1" in robot:
        # states = torch.cat([states[..., :23], states[..., 37:60], states[..., 76:77], states[..., 81:]], axis=-1)
        pass
    elif "go2" in robot or "a1" in robot or "anymal-D" in robot:
        # For Go2, use all 37 dimensions (12 joint pos + 12 joint vel + 13 root state)
        pass  # Keep all dimensions
        
    if normalize and state_mean is not None and state_std is not None:
        states = ((states-state_mean)/state_std).squeeze(0)
    
    # Convert to double precision to match Koopman model
    return states.double() 

def compute_dof_acc(dof_vel, last_vel, dt):
    """Compute joint acceleration from velocity difference."""
    dof_acc = (dof_vel - last_vel) / dt
    return dof_acc

def main():
    global ref_data, x_ref, robot, ref_dof_acc, env_cfg

    # Determine robot type from task name
    if "g1" in args_cli.task.lower():
        robot = "g1"
        # G1 ref_data placeholder - replace with actual path
        ref_path_format = "../data/g1_flat/reference_repository/{}.npz"
        ref_filename = "2025-03-19_17-35-32_trajnum3000_trajlen500"
    elif "go2" in args_cli.task.lower():
        robot = "go2"
        # Go2 ref_data placeholder - replace with actual path
        ref_path_format = "../data/unitree_go2_flat/reference_repository/{}.npz"
        ref_filename = "None_trajnum3000_trajlen500"
    else:
        assert False, "Only G1 and Go2 robots are supported."
    
    # Determine if rough terrain
    if "rough" in args_cli.task.lower():
        robot += "_rough"
    
    # Check if a direct model path was provided
    if args_cli.model_path:
        model_path = args_cli.model_path
        print(f"Using direct model path: {model_path}")
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return
        # Use default parameters for direct model path
        model_params_from_csv = None
    # Check if a specific model run was requested
    elif args_cli.load_run or os.path.exists(args_cli.csv_log_path):
        # Get model info from CSV for the specified environment
        csv_log_path = args_cli.csv_log_path
        print(f"Loading model info from CSV: {csv_log_path}")
        
        model_info = find_model_paths_from_csv(csv_log_path, "G1" if "g1" in robot else "Go2")
        
        if model_info is None or model_info.empty:
            print(f"No models found for {robot} in CSV file {csv_log_path}")
            print("Available robot types in CSV:")
            # Try to show what robot types are available
            try:
                import pandas as pd
                df = pd.read_csv(csv_log_path)
                if 'env_name' in df.columns:
                    available_envs = df['env_name'].unique()
                    print(f"  {list(available_envs)}")
            except Exception as e:
                print(f"  Could not read CSV file: {e}")
            return
            
        # Extract model path and parameters from CSV
        model_path = model_info['model_path']
        model_params_from_csv = model_info  # Contains all parameters including encode_dim
        print(f"Using model: {model_path}")
        
        # Print available parameters from CSV
        print("[INFO] Model parameters from CSV:")
        for col in model_info.index:
            if col not in ['model_path', 'env_name']:
                print(f"  {col}: {model_info[col]}")
    else:
        # Use the traditional approach from the original script
        final_log_path = os.path.join("logs", "rsl_rl", args_cli.log_path)
        final_log_path = os.path.abspath(final_log_path)
        resume_path = os.path.join(final_log_path, args_cli.algo, robot)
        
        # Check if the resume path exists
        if not os.path.exists(resume_path):
            print(f"Error: Checkpoint directory does not exist: {resume_path}")
            if args_cli.test_mode:
                print("Running in test mode - creating dummy model...")
                # Create a dummy model for testing
                os.makedirs(resume_path, exist_ok=True)
                dummy_checkpoint_dir = os.path.join(resume_path, "test_run", "checkpoint")
                os.makedirs(dummy_checkpoint_dir, exist_ok=True)
                model_path = os.path.join(dummy_checkpoint_dir, "model1.pt")
                # Create dummy model file
                torch.save({}, model_path)
                print(f"Created dummy model at: {model_path}")
            else:
                print("Please provide a valid checkpoint path or use --load_run with --csv_log_path")
                print("Or add --test_mode to create dummy data for testing")
                print("Example:")
                print("  python mpc_tracking.py --headless --task Isaac-Velocity-Flat-Unitree-Go2-v0 --algo incremental --load_run true")
                return
            
        if args_cli.load_run is None:
            log_dirs = os.listdir(resume_path)
            if not log_dirs:
                print(f"Error: No log directories found in {resume_path}")
                return
            log_file = log_dirs[0]
        else:
            log_file = args_cli.load_run
            
        resume_path = os.path.join(resume_path, log_file, "checkpoint")
        
        if not os.path.exists(resume_path):
            print(f"Error: Checkpoint directory does not exist: {resume_path}")
            return
            
        checkpoint_files = [f for f in os.listdir(resume_path) if f.startswith("model") and f.endswith(".pt")]
        if not checkpoint_files:
            print(f"Error: No model checkpoint files found in {resume_path}")
            return
            
        last_checkpoint = "model"+str(max([int(i.split("model")[1].split(".pt")[0]) for i in checkpoint_files]))+".pt"
        
        if args_cli.checkpoint:
            model_path = os.path.join(resume_path, args_cli.checkpoint)
        else:
            model_path = os.path.join(resume_path, last_checkpoint)
    
    # Load parameters - prioritize CSV parameters if available
    param_path = model_path.split("checkpoint")[0] + "param.yaml" if "checkpoint" in model_path else model_path.replace(".pth", "_params.yaml")
    
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            params = yaml.safe_load(f)
    else:
        # Use default parameters if param file doesn't exist
        print(f"Parameter file not found at {param_path}, using defaults")
        params = {
            "action_dim": 23 if "g1" in robot else 12,
            "state_dim": 87 if "g1" in robot else 37,
            "normalize": True,
            "hidden_dim": 256,
            "num_blocks": 3,
            "encode_dim": 64,
            "N_koopman": 64 + (87 if "g1" in robot else 37)
        }
    
    # Override with CSV parameters if available
    if model_params_from_csv is not None:
        import pandas as pd  # Import pandas for null checking
        print("[INFO] Overriding parameters with CSV values:")
        for param_name in ['encode_dim', 'hidden_dim', 'num_blocks', 'state_dim', 'action_dim']:
            if param_name in model_params_from_csv.index and not pd.isna(model_params_from_csv[param_name]):
                old_value = params.get(param_name, 'Not set')
                params[param_name] = int(model_params_from_csv[param_name])  # Convert to int
                print(f"  {param_name}: {old_value} -> {params[param_name]}")
        
        # Handle special parameter mappings from CSV to our parameter names
        if 'u_dim' in model_params_from_csv.index and not pd.isna(model_params_from_csv['u_dim']):
            old_value = params.get('action_dim', 'Not set')
            params['action_dim'] = int(model_params_from_csv['u_dim'])
            print(f"  action_dim (from u_dim): {old_value} -> {params['action_dim']}")
            
        if 'hidden_layers' in model_params_from_csv.index and not pd.isna(model_params_from_csv['hidden_layers']):
            old_value = params.get('num_blocks', 'Not set')
            params['num_blocks'] = int(model_params_from_csv['hidden_layers'])
            print(f"  num_blocks (from hidden_layers): {old_value} -> {params['num_blocks']}")

    u_dim = params["action_dim"]
    s_dim = params["state_dim"]
    normalize = params.get("normalize", True)
    hidden_dim = params.get("hidden_dim", 256)
    num_blocks = params.get("num_blocks", 3)
    encode_dim = params.get("encode_dim", 64)
    traj_len = 23 if "h1" in robot else 15

    # Load the appropriate network based on algorithm
    if args_cli.algo == "incremental" or args_cli.algo == "dkuc": 
        N_koopman = params.get("N_koopman", s_dim + encode_dim)
        print(f"[INFO] State dim: {s_dim}, Encode dim: {encode_dim}, N_koopman: {N_koopman}")
        net = load_koopman_model(model_path, s_dim, u_dim, device)
    elif args_cli.algo == "dkac":
        N_koopman = params.get("N_koopman", s_dim + encode_dim)
        print(f"[INFO] State dim: {s_dim}, Encode dim: {encode_dim}, N_koopman: {N_koopman}")
        net = load_koopman_model(model_path, s_dim, u_dim, device)
    elif args_cli.algo == "gmm":
        N_koopman = encode_dim + s_dim
        num_gmm = params.get("num_gmm", 5)
        print(f"[INFO] State dim: {s_dim}, Encode dim: {encode_dim}, N_koopman: {N_koopman}")
        net = load_koopman_model(model_path, s_dim, u_dim, device)
    elif args_cli.algo == "nndm":
        print(f"[INFO] NNDM - State dim: {s_dim}")
        net = load_koopman_model(model_path, s_dim, u_dim, device)
    else:
        assert False, "Algorithm not supported."

    net.eval()
    
    # Test the network to understand its actual encoded dimension
    test_input = torch.randn(1, s_dim).double().to(device)
    with torch.no_grad():
        test_encoded = net.encode(test_input)
        actual_koopman_dim = test_encoded.shape[-1]
        print(f"[INFO] Actual Koopman dimension from network: {actual_koopman_dim}")
        
        # Verify our parameters are correct
        if actual_koopman_dim != N_koopman:
            print(f"[WARNING] Expected N_koopman ({N_koopman}) != Actual ({actual_koopman_dim})")
            print(f"[INFO] Using actual dimension: {actual_koopman_dim}")
            N_koopman = actual_koopman_dim

    # Determine number of environments first
    num_envs = int(args_cli.ref.split("trajnum")[1]) if "trajnum" in args_cli.ref else 500
    print(f"[INFO] Number of environments requested: {num_envs}")

    # Load reference trajectory
    ref_path = ref_path_format.format(ref_filename)
    print(f"Loading reference trajectory from: {ref_path}")
    
    if os.path.exists(ref_path):
        ref_data_np = np.load(ref_path, allow_pickle=True)
        state_data = torch.DoubleTensor(ref_data_np["state_data"]).to(device)
        action_data = torch.DoubleTensor(ref_data_np["action_data"]).to(device)
        
        # Limit to the number of environments we're using
        print(f"[INFO] Original trajectory shape: {state_data.shape}")
        available_trajs = state_data.shape[1]
        num_envs = min(num_envs, available_trajs)
        print(f"[INFO] Using {num_envs} environments (limited by available trajectories)")
        
        state_data = state_data[:, :num_envs, :]
        action_data = action_data[:, :num_envs, :]
        print(f"[INFO] Limited trajectory shape: {state_data.shape}")
        
        ref_data = OrderedDict({"state_data": state_data, "action_data": action_data})
    else:
        print(f"Warning: Reference data file not found at {ref_path}")
        # Create dummy reference data for testing
        if "g1" in robot:
            state_dim = 87 + 12  # g1 has 87 state dimensions
            action_dim = 23
        else:  # go2
            state_dim = 37 + 12  # go2 has 37 state dimensions
            action_dim = 12
            
        state_data = torch.zeros((15, num_envs, state_dim)).double().to(device)
        action_data = torch.zeros((14, num_envs, action_dim)).double().to(device)
        ref_data = OrderedDict({"state_data": state_data, "action_data": action_data})
    
    x_ref = state_data.clone()
    u_ref = action_data.clone()
    
    if "g1" in robot:
        # x_ref = torch.cat([x_ref[..., :23], x_ref[..., 37:60], x_ref[..., 76:77], x_ref[..., 81:]], axis=-1)
        x_ref = x_ref#torch.cat([...], axis=-1)
        u_ref = u_ref#[..., :23]
    elif "go2" in robot or "a1" in robot or "anymal-D" in robot:
        # For Go2, the reference data should already be 37 dimensions
        if state_data.shape[-1] < 37:
            # The reference data might be missing some dimensions
            missing_dims = 37 - state_data.shape[-1]
            padding = torch.zeros((*state_data.shape[:-1], missing_dims)).double().to(device)
            x_ref = torch.cat([state_data, padding], dim=-1)
            print(f"[INFO] Padded reference trajectory from {state_data.shape[-1]} to {x_ref.shape[-1]} dimensions")
        elif state_data.shape[-1] == 37:
            # Perfect match, use as is
            x_ref = state_data.clone()
        else:
            # Reference data has more dimensions, need to select the right ones
            x_ref = torch.cat([x_ref[..., :24], x_ref[..., 26:]], axis=-1)
            
        u_ref = u_ref
    else:
        # Default handling - should match one of the above conditions
        pass
        
    print(f"[INFO] Reference trajectory shape: {x_ref.shape}")
    print(f"[INFO] Reference actions shape: {u_ref.shape}")
    
    action_l = u_ref.min(dim=0).values
    action_u = u_ref.max(dim=0).values
    
    # Load normalization data if needed
    state_mean, state_std = None, None
    if normalize:
        normalize_data_path = model_path.split("checkpoint")[0] + "normalize_data.npz" if "checkpoint" in model_path else model_path.replace(".pth", "_normalize.npz")
        
        if os.path.exists(normalize_data_path):
            normalize_data = np.load(normalize_data_path)
            state_mean = torch.tensor(normalize_data["state_mean"], dtype=torch.double).to(device)
            state_std = torch.tensor(normalize_data["state_std"], dtype=torch.double).to(device)
            x_ref = ((x_ref-state_mean)/state_std).squeeze(0)
        else:
            print(f"Normalization data not found at {normalize_data_path}, continuing without normalization")
            normalize = False

    # Setup environment
    env_cfg = parse_env_cfg(args_cli.task, num_envs=num_envs)
    
    # Define event terms for resetting the robot
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

    env_cfg.terminations.base_contact = None
    if args_cli.viewer:
        eye = tuple(args_cli.viewer[:3])
        lookat = tuple(args_cli.viewer[3:])
        env_cfg.viewer = ViewerCfg(eye=eye, lookat=lookat, origin_type="world", resolution=(3840, 2160))
    env_cfg.commands.base_velocity.debug_vis = False
    
    # Set environment parameters
    env_cfg.num_envs = x_ref.shape[1]
    env_cfg.episode_length_s = (x_ref.shape[0] - traj_len - 1) * env_cfg.sim.dt * env_cfg.decimation
    
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    # Compute reference DoF acceleration
    ref_dof_acc = compute_ref_dof_acc()
    
    # Video recording wrapper
    if args_cli.video:
        # Create log directory
        log_root_path = os.path.join("logs", "rsl_rl", "koopman", robot)
        log_root_path = os.path.abspath(log_root_path)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = f"{args_cli.algo}-{robot}-" + log_dir
        log_dir = os.path.join(log_root_path, log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    
    # Wrap environment for rsl-rl
    env = RslRlVecEnvWrapper(env)
    
    max_len = x_ref.shape[0] - traj_len - 1
    num_steps = args_cli.num_steps_per_env if args_cli.num_steps_per_env <= max_len else max_len

    # Initialize metrics tracking
    JrPE_list = [[] for _ in range(env.num_envs)]
    JrVE_list = [[] for _ in range(env.num_envs)]
    JrAE_list = [[] for _ in range(env.num_envs)]
    RPE_list = [[] for _ in range(env.num_envs)]
    ROE_list = [[] for _ in range(env.num_envs)]
    RLVE_list = [[] for _ in range(env.num_envs)]
    RAVE_list = [[] for _ in range(env.num_envs)]
    
    survival_steps = [0.0 for _ in range(env.num_envs)]
    collect_done = [False for _ in range(env.num_envs)]

    # Start tracking
    start_time = time.time()
    obs = get_obs(env, normalize, state_mean, state_std)
    
    # Get base environment for data access
    if hasattr(env, 'unwrapped'):
        base_env = env.unwrapped
    elif hasattr(env, 'env'):
        base_env = env.env
    else:
        base_env = env
        
    last_vel = base_env.scene['robot'].data.joint_vel.clone()
    dones = [False for _ in range(env.num_envs)]
    
    dt = env_cfg.sim.dt * env_cfg.decimation
    
    with torch.inference_mode():
        for i in tqdm(range(num_steps)):
            # Get current state
            robot_data = base_env.scene['robot'].data
            dof_pos = robot_data.joint_pos.clone()
            dof_vel = robot_data.joint_vel.clone()
            root_state = robot_data.root_state_w.clone()
            state = torch.cat([dof_pos, dof_vel, root_state], axis=1)

            # Check the correctness of initialization
            for j in range(env.num_envs):
                if base_env.episode_length_buf[j] == 0:
                    # x, y don't follow ref_data (except Rough terrain)
                    assert torch.abs(state[j, :-13] - ref_data["state_data"][0, j, :-13]).mean() == 0.0, f"State mismatch"
                    assert torch.abs(state[j, -11:] - ref_data["state_data"][0, j, -11:]).mean() == 0.0, f"State mismatch"

            # Compute tracking metrics
            for j in range(env.num_envs):
                if i != 0 and dones[j]:
                    last_vel[j] = dof_vel[j].clone()

                if "g1" in robot:
                    JrPE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, :23] - state[j, :23]).mean())
                    JrVE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 37:60] - state[j, 37:60]).mean())
                    JrAE_list[j].append(torch.abs(ref_dof_acc[base_env.episode_length_buf[j], j, :] - compute_dof_acc(dof_vel, last_vel, dt)[j, :23]).mean())
                    RPE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 74:77] - state[j, 74:77]).mean())
                    ROE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 77:81] - state[j, 77:81]).mean())
                    RLVE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 81:84] - state[j, 81:84]).mean())
                    RAVE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 84:87] - state[j, 84:87]).mean())
                elif "go2" in robot or "a1" in robot or "anymal-D" in robot:
                    JrPE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, :12] - state[j, :12]).mean())
                    JrVE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 12:24] - state[j, 12:24]).mean())
                    JrAE_list[j].append(torch.abs(ref_dof_acc[base_env.episode_length_buf[j], j, :] - compute_dof_acc(dof_vel, last_vel, dt)[j, :]).mean())
                    RPE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 24:27] - state[j, 24:27]).mean())
                    ROE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 27:31] - state[j, 27:31]).mean())
                    RLVE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 31:34] - state[j, 31:34]).mean())
                    RAVE_list[j].append(torch.abs(ref_data["state_data"][base_env.episode_length_buf[j], j, 34:37] - state[j, 34:37]).mean())
                
            cur_JrPE = torch.tensor([env_list[-1] if env_list else 0.0 for env_list in JrPE_list])

            # Calculate MPC control actions based on algorithm
            range_matrix = (base_env.episode_length_buf).unsqueeze(0) + torch.arange(traj_len+1).unsqueeze(1).to(device)
            sliced_x_ref = x_ref[range_matrix, (torch.arange(base_env.episode_length_buf.shape[0])).unsqueeze(0)]
            
            # Ensure x_ref is double precision to match model
            sliced_x_ref = sliced_x_ref.double()
            
            if args_cli.algo == "incremental" or args_cli.algo == "dkuc":
                actions = Torch_MPC(net, obs, sliced_x_ref, traj_len, u_dim, N_koopman, device, robot)
            elif args_cli.algo == "dkac":
                actions = Torch_MPC_DKAC(net, obs, sliced_x_ref, traj_len, u_dim, N_koopman, device, robot)
            elif args_cli.algo == "gmm":
                actions_list, res_list = [], []
                for gmm_idx in range(num_gmm):
                    actions, res = Torch_MPC_GMM(net, obs, sliced_x_ref, traj_len, u_dim, N_koopman, device, robot, gmm_idx)
                    actions_list.append(actions)
                    res_list.append(res)
                gmm_actions = torch.stack(actions_list, dim=0)
                gmm_res = torch.stack(res_list, dim=0)
                best_idx = torch.argmin(gmm_res, dim=0)
                actions = gmm_actions[best_idx, torch.arange(env.num_envs)]
            elif args_cli.algo == "nndm":
                actions = NMPC(net, obs, sliced_x_ref, traj_len, u_dim, s_dim, device, robot, action_l, action_u)

            # Clamp actions to valid range and convert back to float for environment
            actions = actions.clamp(action_l.double(), action_u.double()).float()
            actions = torch.cat([actions, torch.zeros(env.num_envs, ref_data["action_data"].shape[-1] - u_ref.shape[-1]).to(device)], axis=-1)

            # Step environment
            _, _, dones, _ = env.step(actions)
            obs = get_obs(env, normalize, state_mean, state_std)
            last_vel = dof_vel.clone()

            # Detect tracking failures
            threshold = 0.2
            if robot == "anymal-D":
                threshold = 0.18 
            elif robot == "a1":  
                threshold = 0.16 
            elif robot == "go2": 
                threshold = 0.16 
            elif robot == "go2_rough": 
                threshold = 0.12 
            elif robot == "h1":
                threshold = 0.15
            elif robot == "g1": 
                threshold = 0.1
            elif robot == "g1_rough": 
                threshold = 0.08

            for j in range(env.num_envs):
                if cur_JrPE[j] > threshold:
                    collect_done[j] = True
                if not collect_done[j]:
                    survival_steps[j] += 1.0
    
    end_time = time.time()
    
    # Calculate metrics
    for i in range(env.num_envs):
        if JrPE_list[i]:  # Check if list is not empty
            JrPE_list[i] = torch.tensor(JrPE_list[i]).mean().item()
            JrVE_list[i] = torch.tensor(JrVE_list[i]).mean().item()
            JrAE_list[i] = torch.tensor(JrAE_list[i]).mean().item()
            RPE_list[i] = torch.tensor(RPE_list[i]).mean().item()
            ROE_list[i] = torch.tensor(ROE_list[i]).mean().item()
            RLVE_list[i] = torch.tensor(RLVE_list[i]).mean().item()
            RAVE_list[i] = torch.tensor(RAVE_list[i]).mean().item()
        else:
            # Handle empty list case
            JrPE_list[i] = 0.0
            JrVE_list[i] = 0.0
            JrAE_list[i] = 0.0
            RPE_list[i] = 0.0
            ROE_list[i] = 0.0
            RLVE_list[i] = 0.0
            RAVE_list[i] = 0.0

    JrPE = sum(JrPE_list) / len(JrPE_list) if JrPE_list else 0.0
    JrVE = sum(JrVE_list) / len(JrVE_list) if JrVE_list else 0.0
    JrAE = sum(JrAE_list) / len(JrAE_list) if JrAE_list else 0.0
    RPE = sum(RPE_list) / len(RPE_list) if RPE_list else 0.0
    ROE = sum(ROE_list) / len(ROE_list) if ROE_list else 0.0
    RLVE = sum(RLVE_list) / len(RLVE_list) if RLVE_list else 0.0
    RAVE = sum(RAVE_list) / len(RAVE_list) if RAVE_list else 0.0
    survival_steps_avg = sum(survival_steps) / len(survival_steps) if survival_steps else 0.0

    metrics = {
        "JrPE": JrPE,
        "JrVE": JrVE,
        "JrAE": JrAE,
        "RPE": RPE,
        "ROE": ROE,
        "RLVE": RLVE,
        "RAVE": RAVE,
        "survival_steps": survival_steps_avg,
        "time": end_time - start_time
    }
    print(metrics)

    # Save metrics
    save_path = os.path.join("logs", "rsl_rl", args_cli.save_path, args_cli.algo, robot)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    # Use model name in the save file
    model_name = os.path.basename(model_path).split('.')[0]
    np.savez(os.path.join(save_path, f"{model_name}_tracking_error.npz"), **metrics)

    # Close the environment
    env.close()

if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()