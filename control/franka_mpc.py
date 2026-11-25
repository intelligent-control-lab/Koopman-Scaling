#!/usr/bin/env python3
"""
MPC Control for Franka Robot Arm
================================

This script implements Model Predictive Control (MPC) using Koopman dynamics
for Franka robot end-effector trajectory tracking.

Trajectories:
- Figure-8 (Lemniscate of Bernoulli)
- 5-point Star

Key Features:
- Receding horizon MPC with trajectory preview
- End-effector position tracking in Koopman space
- Feedforward compensation option
- Multiple metrics: tracking error, max error, path completion

Usage:
    python franka_mpc.py --project_name Aug_8
    python franka_mpc.py --project_name FrankaImproved

Author: Koopman Scaling Law Analysis
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import scipy.linalg
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm

sys.path.append("../utility")
from network import KoopmanNet

try:
    import pybullet as pb
    from franka_env import FrankaEnv
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("Warning: PyBullet not available. Franka evaluation will be skipped.")


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class FrankaMPCConfig:
    """MPC configuration for Franka robot."""
    # MPC settings
    horizon: int = 15

    # Cost weights (in Koopman space)
    Q_position: float = 10.0     # Weight on xyz position error
    Q_joint: float = 1.0         # Weight on joint position error
    Q_latent: float = 0.001      # Weight on latent states
    R_weight: float = 0.01       # Control effort weight

    # Evaluation
    warmup_steps: int = 100      # Skip initial transient
    noise_sigma: float = 0.01    # Observation noise

    # Action clipping
    action_clip: Tuple[float, float] = (-0.3, 0.3)

    # Stability
    stability_threshold: float = 1.10


# Franka environment info
# Training uses joint-only state: [q(7), dq(7)] = 14 dims
# franka_env.py returns [xyz(3), quat(4), q(7), dq(7)] = 21 dims
# We need to extract the last 14 elements for the Koopman model
FRANKA_CONFIG = {
    'state_dim': 14,   # joint_pos(7) + joint_vel(7)
    'full_state_dim': 21,  # xyz(3) + quat(4) + joint_pos(7) + joint_vel(7)
    'action_dim': 7,
    'dt': 0.02,
}


# =============================================================================
# Trajectory Generators
# =============================================================================
def generate_figure8_trajectory(T: int, dt: float = 0.02) -> np.ndarray:
    """
    Generate figure-8 (Lemniscate of Bernoulli) trajectory.

    Args:
        T: Number of time steps
        dt: Time step

    Returns:
        positions: (T, 3) end-effector xyz positions
    """
    t = 1.6 + dt * np.arange(T)
    a = 0.3

    x = 0.3 * np.ones(T)
    y = a * np.cos(t) / (1 + np.sin(t)**2)
    z = 0.59 + 2 * a * np.sin(t) * np.cos(t) / (1 + np.sin(t)**2)

    return np.column_stack([x, y, z])


def generate_star_trajectory(T: int, dt: float = 0.02) -> np.ndarray:
    """
    Generate 5-point star trajectory.

    Args:
        T: Number of time steps
        dt: Time step

    Returns:
        positions: (T, 3) end-effector xyz positions
    """
    center = np.array([0.0, 0.6])
    radius = 0.3
    theta_ = np.pi / 10.0
    eradius = np.tan(2*theta_) * radius * np.cos(theta_) - radius * np.sin(theta_)

    # Star vertices
    Star = np.zeros((11, 2))
    for i in range(5):
        th = 2*np.pi/5 * (i + 0.25)
        be = 2*np.pi/5 * (i + 0.75)
        Star[2*i] = [np.cos(th)*radius + center[0], np.sin(th)*radius + center[1]]
        Star[2*i+1] = [np.cos(be)*eradius + center[0], np.sin(be)*eradius + center[1]]
    Star[-1] = Star[0]

    # Interpolate between vertices
    each_num = max(1, (T - 10) // 10)
    refs = np.zeros((T, 2))
    idx = 0
    for i in range(10):
        if idx >= T:
            break
        refs[idx] = Star[i]
        idx += 1
        for j in range(each_num):
            if idx >= T:
                break
            tau = (j+1) / (each_num+1)
            refs[idx] = tau*Star[i+1] + (1-tau)*Star[i]
            idx += 1

    # Fill remaining
    refs[idx:] = Star[-1]

    x = 0.3 * np.ones(T)
    y = refs[:, 0]
    z = refs[:, 1]

    return np.column_stack([x, y, z])


# =============================================================================
# Utility Functions
# =============================================================================
def spectral_radius(A: np.ndarray) -> float:
    """Compute spectral radius of matrix A."""
    return float(np.max(np.abs(scipy.linalg.eigvals(A))))


def extract_joint_state(full_state: np.ndarray) -> np.ndarray:
    """
    Extract joint-only state from full FrankaEnv state.

    FrankaEnv.get_state() returns [xyz(3), quat(4), q(7), dq(7)] = 21 dims
    Training uses [q(7), dq(7)] = 14 dims

    Args:
        full_state: Full state from FrankaEnv (21,)

    Returns:
        Joint state [q(7), dq(7)] (14,)
    """
    # xyz = full_state[0:3]
    # quat = full_state[3:7]
    q = full_state[7:14]   # joint positions
    dq = full_state[14:21]  # joint velocities
    return np.concatenate([q, dq])


def get_ee_position(env) -> np.ndarray:
    """Get end-effector xyz position from environment."""
    ee_state = pb.getLinkState(env.robot, env.ee_id)
    return np.array(ee_state[4])  # World position


def load_koopman_model(model_path: str, state_dim: int, action_dim: int,
                       encode_dim: int, use_residual: bool = True,
                       device: str = 'cpu'
                       ) -> Tuple[KoopmanNet, np.ndarray, np.ndarray, dict]:
    """Load Koopman model and extract dynamics matrices.

    Returns:
        net: KoopmanNet model
        A: Koopman A matrix
        B: Koopman B matrix
        norm_stats: Normalization statistics (or None if not used)
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        layers = checkpoint.get('layer', None)
    else:
        state_dict = checkpoint
        layers = None

    # Extract norm_stats if available
    norm_stats = checkpoint.get('norm_stats', None)

    if layers is None:
        if 'lA.weight' in state_dict:
            NKoopman = state_dict['lA.weight'].shape[0]
            enc_dim = NKoopman - state_dim
            layers = [state_dim, 256, 256, 256, enc_dim]
        else:
            layers = [state_dim, 256, 256, 256, encode_dim]

    NKoopman = layers[0] + layers[-1]
    net = KoopmanNet(layers, NKoopman, action_dim, use_residual)
    net.load_state_dict(state_dict)

    # Move to device and set dtype
    net.to(device).double().eval()

    A = state_dict['lA.weight'].cpu().numpy()
    B = state_dict['lB.weight'].cpu().numpy()

    return net, A, B, norm_stats


def normalize_state(state: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Normalize state using training statistics."""
    if norm_stats is None:
        return state
    state_mean = np.array(norm_stats['state_mean'])
    state_std = np.array(norm_stats['state_std'])
    return (state - state_mean) / state_std


def denormalize_action(action: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Denormalize action to physical units."""
    if norm_stats is None:
        return action
    action_mean = np.array(norm_stats['action_mean'])
    action_std = np.array(norm_stats['action_std'])
    return action * action_std + action_mean


def ik_to(env, x, y, z, threshold=1e-6, max_iter=10000):
    """Compute inverse kinematics to reach target position."""
    it, close = 0, False
    x, y, z = float(x), float(y), float(z)
    while not close and it < max_iter:
        joint_poses = pb.calculateInverseKinematics(env.robot, env.ee_id, [x, y, z])
        for j in range(7):
            pb.resetJointState(env.robot, j, float(joint_poses[j]))
        new_pos = pb.getLinkState(env.robot, env.ee_id)[4]
        dx, dy, dz = (x - new_pos[0]), (y - new_pos[1]), (z - new_pos[2])
        close = (dx*dx + dy*dy + dz*dz) < threshold
        it += 1
    return joint_poses[:7]


# =============================================================================
# Koopman MPC Controller for Trajectory Tracking
# =============================================================================
class FrankaMPC:
    """
    Model Predictive Control for Franka trajectory tracking.

    Solves:
        min  sum_{k=0}^{H} ||z_k - z_ref_k||_Q^2 + sum_{k=0}^{H-1} ||u_k||_R^2
        s.t. z_{k+1} = A z_k + B u_k

    With time-varying reference trajectory z_ref_k.
    """

    def __init__(self, net: KoopmanNet, A: np.ndarray, B: np.ndarray,
                 config: FrankaMPCConfig, norm_stats: dict = None,
                 device: str = 'cpu'):
        """Initialize MPC controller.

        Args:
            net: KoopmanNet model
            A: Koopman A matrix
            B: Koopman B matrix
            config: MPC configuration
            norm_stats: Normalization statistics (state_mean, state_std, action_mean, action_std)
            device: torch device ('cpu' or 'cuda')
        """
        self.net = net
        self.A = np.asarray(A, dtype=np.float64)
        self.B = np.asarray(B, dtype=np.float64)
        self.H = config.horizon
        self.config = config
        self.norm_stats = norm_stats
        self.device = device

        self.latent_dim = A.shape[0]
        self.action_dim = B.shape[1]
        self.state_dim = FRANKA_CONFIG['state_dim']

        # Build cost matrices
        self.Q = self._build_Q()
        self.R = np.zeros((self.action_dim, self.action_dim))  # R = 0 (match mpc_tracking.py)

        # Precompute MPC matrices
        self._precompute_matrices()

    def _build_Q(self) -> np.ndarray:
        """Build state cost matrix for joint-only state [q(7), dq(7)].

        Zero cost on latent states (matching mpc_tracking.py).
        """
        Q = np.zeros((self.latent_dim, self.latent_dim))

        # Joint positions (indices 0-6): high weight for tracking
        for i in range(min(7, self.latent_dim)):
            Q[i, i] = self.config.Q_position

        # Joint velocities (indices 7-13): lower weight
        for i in range(7, min(14, self.latent_dim)):
            Q[i, i] = self.config.Q_joint

        # Latent states: ZERO cost (matching mpc_tracking.py)
        # Q[self.state_dim:, self.state_dim:] = 0.0  (already zero)

        return Q

    def _precompute_matrices(self):
        """Precompute block matrices for efficient MPC solve."""
        H = self.H
        n = self.latent_dim
        m = self.action_dim

        # State propagation matrix
        self.M = np.vstack([np.linalg.matrix_power(self.A, i) for i in range(H + 1)])

        # Control influence matrix
        self.C = np.zeros(((H + 1) * n, H * m))
        for i in range(1, H + 1):
            for j in range(i):
                self.C[i*n:(i+1)*n, j*m:(j+1)*m] = (
                    np.linalg.matrix_power(self.A, i-j-1) @ self.B
                )

        # Terminal cost F (same as Q, matching mpc_tracking.py structure)
        F = self.Q.copy()

        # Block diagonal cost matrices with terminal cost
        Qh = scipy.linalg.block_diag(*([self.Q] * H), F)  # [Q,Q,...,Q,F]
        Rh = scipy.linalg.block_diag(*([self.R] * H))

        self.P = 2 * (Rh + self.C.T @ Qh @ self.C)
        self.P_inv = np.linalg.inv(0.5 * self.P)
        self.Qh = Qh

    def encode(self, state: np.ndarray) -> np.ndarray:
        """Encode state to Koopman latent space.

        If normalization stats are available, normalizes the state first.
        Uses torch.inference_mode() for faster inference.
        """
        # Normalize state if norm_stats available
        state_norm = normalize_state(state, self.norm_stats)

        with torch.inference_mode():
            s_tensor = torch.tensor(state_norm.reshape(1, -1), dtype=torch.float64, device=self.device)
            z = self.net.encode(s_tensor).cpu().numpy().flatten()
        return z

    def solve(self, current_state: np.ndarray,
              ref_trajectory: np.ndarray) -> np.ndarray:
        """
        Solve MPC with trajectory reference.

        Args:
            current_state: Current state in PHYSICAL units (state_dim,)
            ref_trajectory: Reference states in PHYSICAL units over horizon (H+1, state_dim)

        Returns:
            u: Optimal control input in PHYSICAL units (action_dim,)
        """
        # Encode current state (normalization handled inside encode)
        z_current = self.encode(current_state)

        # Encode reference trajectory (normalization handled inside encode)
        Z_ref = np.zeros((self.H + 1) * self.latent_dim)
        for i in range(min(len(ref_trajectory), self.H + 1)):
            z_ref_i = self.encode(ref_trajectory[i])
            Z_ref[i*self.latent_dim:(i+1)*self.latent_dim] = z_ref_i

        # Fill remaining with last reference
        if len(ref_trajectory) < self.H + 1:
            z_last = self.encode(ref_trajectory[-1])
            for i in range(len(ref_trajectory), self.H + 1):
                Z_ref[i*self.latent_dim:(i+1)*self.latent_dim] = z_last

        # QP solution (no feedforward, matching mpc_tracking.py)
        predicted = self.M @ z_current
        q = 2 * (predicted - Z_ref) @ self.Qh @ self.C
        U_opt = -0.5 * self.P_inv @ q.T

        # Extract first control (in normalized space)
        u_norm = U_opt[:self.action_dim]

        # Denormalize action to physical units
        u = denormalize_action(u_norm, self.norm_stats)

        return u


# =============================================================================
# Control Evaluation
# =============================================================================
@dataclass
class FrankaControlResult:
    """Results from Franka control evaluation."""
    error_figure8: float = float('inf')
    error_star: float = float('inf')
    max_error_figure8: float = float('inf')
    max_error_star: float = float('inf')
    max_eigenvalue: float = float('inf')
    stable: bool = False


def evaluate_franka_mpc(model_path: str, encode_dim: int,
                        config: FrankaMPCConfig = None,
                        device: str = 'cpu') -> FrankaControlResult:
    """
    Evaluate MPC control for Franka robot.

    Args:
        model_path: Path to trained Koopman model
        encode_dim: Encoding dimension (absolute)
        config: MPC configuration
        device: torch device ('cpu' or 'cuda')

    Returns:
        FrankaControlResult with tracking errors
    """
    if not PYBULLET_AVAILABLE:
        return FrankaControlResult()

    if config is None:
        config = FrankaMPCConfig()

    state_dim = FRANKA_CONFIG['state_dim']
    action_dim = FRANKA_CONFIG['action_dim']
    dt = FRANKA_CONFIG['dt']
    action_low, action_high = config.action_clip

    # Load model
    try:
        net, A, B, norm_stats = load_koopman_model(model_path, state_dim, action_dim, encode_dim, device=device)
        if norm_stats is not None:
            print(f"  Using normalization (state_mean shape: {len(norm_stats['state_mean'])})")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return FrankaControlResult()

    # Check stability
    rho_A = spectral_radius(A)
    if rho_A > config.stability_threshold:
        print(f"  Unstable dynamics: rho(A) = {rho_A:.4f}")
        return FrankaControlResult(max_eigenvalue=rho_A)

    # Create MPC controller with normalization stats
    mpc = FrankaMPC(net, A, B, config, norm_stats=norm_stats, device=device)

    # Create environment
    try:
        env = FrankaEnv(render=False)
    except Exception as e:
        print(f"  Error creating Franka env: {e}")
        return FrankaControlResult()

    def run_trajectory(traj_positions: np.ndarray, traj_name: str) -> Tuple[float, float]:
        """Run a trajectory and compute tracking error."""
        T = len(traj_positions)
        env.reset()

        # Compute joint positions via IK for full reference trajectory
        # ref_states: [q(7), dq(7)] = 14 dims (joint-only, matching training)
        ref_states = np.zeros((T, state_dim))

        for i in range(T):
            joints = ik_to(env, traj_positions[i, 0], traj_positions[i, 1], traj_positions[i, 2])
            ref_states[i, :7] = joints  # q (joint positions)
            # ref_states[i, 7:14] = 0  # dq (joint velocities, zero at reference)

        # Initialize at first reference (slightly offset)
        joints_init = ik_to(env, traj_positions[0, 0],
                           traj_positions[0, 1] + 0.01,
                           traj_positions[0, 2] + 0.043)
        for j, jnt in enumerate(joints_init):
            pb.resetJointState(env.robot, j, float(jnt))

        # Get initial state (21 dims from FrankaEnv)
        full_state = env.get_state()
        tracked_positions = []
        errors = []

        for step in range(T - config.horizon - 1):
            # Extract joint-only state for Koopman model (14 dims)
            joint_state = extract_joint_state(full_state)

            # Add observation noise to joint positions
            noise = np.random.randn(7) * config.noise_sigma
            joint_state_noisy = joint_state.copy()
            joint_state_noisy[:7] += noise

            # Reference window (joint states)
            ref_window = ref_states[step:step+config.horizon+1]

            # Get MPC control
            try:
                u = mpc.solve(joint_state_noisy, ref_window)
            except Exception as e:
                print(f"  MPC solve error at step {step}: {e}")
                break

            # Clip and apply
            u = np.clip(u, action_low, action_high)
            full_state = env.step(u)

            if step >= config.warmup_steps:
                # Get current end-effector position for error computation
                ee_pos = get_ee_position(env)
                tracked_positions.append(ee_pos.copy())
                # Track y,z error (x is constant)
                err = np.linalg.norm(ee_pos[1:3] - traj_positions[step, 1:3])
                errors.append(err)

        # Compute metrics
        if len(errors) > 0:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
        else:
            mean_error = float('inf')
            max_error = float('inf')

        return mean_error, max_error

    # Evaluate figure-8 trajectory
    steps_fig8 = 300
    fig8_positions = generate_figure8_trajectory(steps_fig8 + config.horizon + 1, dt)
    error_fig8, max_error_fig8 = run_trajectory(fig8_positions, "figure-8")
    print(f"  Figure-8 error: {error_fig8:.4f} (max: {max_error_fig8:.4f})")

    # Evaluate star trajectory
    steps_star = 300
    star_positions = generate_star_trajectory(steps_star + config.horizon + 1, dt)
    error_star, max_error_star = run_trajectory(star_positions, "star")
    print(f"  Star error: {error_star:.4f} (max: {max_error_star:.4f})")

    # Cleanup
    try:
        pb.disconnect()
    except:
        pass

    return FrankaControlResult(
        error_figure8=error_fig8,
        error_star=error_star,
        max_error_figure8=max_error_fig8,
        max_error_star=max_error_star,
        max_eigenvalue=rho_A,
        stable=True
    )


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================
def select_best_models(df: pd.DataFrame, n_best: int = 3,
                       largest_only: bool = False) -> pd.DataFrame:
    """
    Select best N models by test_Kloss for each (train_samples, encode_dim_param, loss_config) combination.

    Args:
        df: DataFrame with model info
        n_best: Number of best models to select per configuration
        largest_only: If True, only select from largest train_samples and encode_dim_param

    Returns:
        Filtered DataFrame with selected models
    """
    loss_configs = [
        (False, False, 'Baseline'),
        (False, True, '+Cov'),
        (True, False, '+Ctrl'),
        (True, True, '+Aux'),
    ]

    if largest_only:
        # Original behavior: filter to largest train_samples and encode_dim_param
        max_samples = df['train_samples'].max()
        df = df[df['train_samples'] == max_samples].copy()
        print(f"  Filtered to train_samples={max_samples}: {len(df)} models")

        max_dim = df['encode_dim_param'].max()
        df = df[df['encode_dim_param'] == max_dim].copy()
        print(f"  Filtered to encode_dim_param={max_dim}: {len(df)} models")

        selected = []
        for use_ctrl, use_cov, label in loss_configs:
            cfg_df = df[
                (df['use_control_loss'] == use_ctrl) &
                (df['use_covariance_loss'] == use_cov)
            ].copy()

            if len(cfg_df) == 0:
                print(f"  {label}: No models found")
                continue

            cfg_df = cfg_df.sort_values('test_Kloss').head(n_best)
            cfg_df['loss_config'] = label
            selected.append(cfg_df)
            print(f"  {label}: Selected {len(cfg_df)} models")

        if not selected:
            return pd.DataFrame()
        return pd.concat(selected, ignore_index=True)

    # New behavior: select across ALL (train_samples, encode_dim_param) combinations
    train_samples_list = sorted(df['train_samples'].unique())
    encode_dim_list = sorted(df['encode_dim_param'].unique())

    print(f"  Train samples: {train_samples_list}")
    print(f"  Encode dims: {encode_dim_list}")

    selected = []
    total_configs = len(train_samples_list) * len(encode_dim_list) * len(loss_configs)
    print(f"  Total configurations to evaluate: {total_configs}")

    for train_samples in train_samples_list:
        for encode_dim_param in encode_dim_list:
            subset = df[
                (df['train_samples'] == train_samples) &
                (df['encode_dim_param'] == encode_dim_param)
            ].copy()

            if len(subset) == 0:
                continue

            for use_ctrl, use_cov, label in loss_configs:
                cfg_df = subset[
                    (subset['use_control_loss'] == use_ctrl) &
                    (subset['use_covariance_loss'] == use_cov)
                ].copy()

                if len(cfg_df) == 0:
                    continue

                # Sort by test_Kloss and take best N
                cfg_df = cfg_df.sort_values('test_Kloss').head(n_best)
                cfg_df['loss_config'] = label
                selected.append(cfg_df)

    if not selected:
        return pd.DataFrame()

    result = pd.concat(selected, ignore_index=True)
    print(f"  Selected {len(result)} models total across all configurations")

    # Print summary by (train_samples, encode_dim_param)
    for train_samples in train_samples_list:
        for encode_dim_param in encode_dim_list:
            count = len(result[
                (result['train_samples'] == train_samples) &
                (result['encode_dim_param'] == encode_dim_param)
            ])
            if count > 0:
                print(f"    samples={train_samples}, dim={encode_dim_param}: {count} models")

    return result


def evaluate_all_models(log_path: str, output_dir: str,
                        config: FrankaMPCConfig = None,
                        n_best: int = 3,
                        largest_only: bool = False,
                        device: str = 'cpu'):
    """
    Evaluate MPC control for Franka models.

    Args:
        log_path: Path to koopman_results_log.csv
        output_dir: Output directory
        config: MPC configuration
        n_best: Number of best models per configuration
        largest_only: If True, only select from largest train_samples and encode_dim_param
        device: torch device ('cpu' or 'cuda')
    """
    if config is None:
        config = FrankaMPCConfig()

    os.makedirs(output_dir, exist_ok=True)

    # Load log
    if not os.path.exists(log_path):
        print(f"Error: Log file not found: {log_path}")
        return

    raw = pd.read_csv(log_path)
    print(f"Loaded {len(raw)} models from {log_path}")

    # Filter to Franka (check both FrankaImproved and Franka)
    franka_df = raw[raw['env_name'].isin(['FrankaImproved', 'Franka'])].copy().reset_index(drop=True)
    print(f"Found {len(franka_df)} Franka models")

    if len(franka_df) == 0:
        print("No Franka models found")
        return

    # Select models
    if largest_only:
        print(f"\nSelecting best {n_best} models from LARGEST configuration only...")
    else:
        print(f"\nSelecting best {n_best} models per (sample_size, model_dim, loss_config)...")
    log = select_best_models(franka_df, n_best, largest_only=largest_only)

    if len(log) == 0:
        print("No models selected")
        return

    print(f"\nEvaluating {len(log)} models total")

    results = []

    for idx in tqdm(range(len(log)), desc="Evaluating Franka models"):
        row = log.iloc[idx]
        model_path = row['model_path']
        encode_dim = int(row['encode_dim'])

        # model_path is relative to control/ directory (e.g., ../log/1-2026-02-04-cuda/...)
        # No need to join with log_path directory
        print(f"\n[{idx+1}/{len(log)}] {model_path}")

        # Verify file exists
        if not os.path.exists(model_path):
            print(f"  Warning: Model file not found: {model_path}")
            continue

        # Evaluate
        result = evaluate_franka_mpc(model_path, encode_dim, config, device=device)

        results.append({
            'model_path': row['model_path'],
            'env_name': 'Franka',
            'train_samples': row.get('train_samples', np.nan),
            'encode_dim': encode_dim,
            'encode_dim_param': row.get('encode_dim_param', np.nan),
            'loss_config': row.get('loss_config', 'Unknown'),
            'use_control_loss': row.get('use_control_loss', False),
            'use_covariance_loss': row.get('use_covariance_loss', False),
            'test_Kloss': row.get('test_Kloss', np.nan),
            'seed': row.get('seed', np.nan),
            'error_figure8': result.error_figure8,
            'error_star': result.error_star,
            'max_error_figure8': result.max_error_figure8,
            'max_error_star': result.max_error_star,
            'max_eigenvalue': result.max_eigenvalue,
            'stable': int(result.stable),
        })

    # Save results
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, 'franka_mpc_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

    # Generate summary
    generate_summary(results_df, output_dir)


def generate_summary(results_df: pd.DataFrame, output_dir: str):
    """Generate summary statistics grouped by (train_samples, encode_dim_param, loss_config)."""

    # Check if loss_config column exists, if not create it
    if 'loss_config' not in results_df.columns:
        def get_loss_config(row):
            use_ctrl = row.get('use_control_loss', False)
            use_cov = row.get('use_covariance_loss', False)
            if use_ctrl and use_cov:
                return '+Aux'
            elif use_ctrl:
                return '+Ctrl'
            elif use_cov:
                return '+Cov'
            else:
                return 'Baseline'
        results_df = results_df.copy()
        results_df['loss_config'] = results_df.apply(get_loss_config, axis=1)

    # Group by (train_samples, encode_dim_param, loss_config)
    group_keys = ['train_samples', 'encode_dim_param', 'loss_config']
    group_keys = [k for k in group_keys if k in results_df.columns]

    summary_rows = []
    for cfg, g in results_df.groupby(group_keys, dropna=False):
        total = len(g)
        stable_cnt = int(g['stable'].sum())

        row = dict(zip(group_keys, cfg if isinstance(cfg, tuple) else [cfg]))
        row.update({
            'n_models': total,
            'n_stable': stable_cnt,
            'stability_rate': stable_cnt / total if total > 0 else 0,
            'mean_error_figure8': g['error_figure8'].replace([np.inf], np.nan).mean(),
            'std_error_figure8': g['error_figure8'].replace([np.inf], np.nan).std(),
            'mean_error_star': g['error_star'].replace([np.inf], np.nan).mean(),
            'std_error_star': g['error_star'].replace([np.inf], np.nan).std(),
            'mean_max_error_figure8': g['max_error_figure8'].replace([np.inf], np.nan).mean(),
            'mean_max_error_star': g['max_error_star'].replace([np.inf], np.nan).mean(),
            'mean_test_Kloss': g['test_Kloss'].mean(),
        })
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, 'franka_mpc_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary to: {summary_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("FRANKA MPC CONTROL SUMMARY")
    print("=" * 70)
    print(f"Total configurations: {len(summary_df)}")
    print(f"Overall mean figure-8 error: {summary_df['mean_error_figure8'].mean():.4f}")
    print(f"Overall mean star error: {summary_df['mean_error_star'].mean():.4f}")
    print(f"Overall mean stability rate: {summary_df['stability_rate'].mean():.2%}")

    # Print results by (train_samples, encode_dim_param) pairs
    if 'train_samples' in summary_df.columns and 'encode_dim_param' in summary_df.columns:
        print("\n" + "-" * 70)
        print("Results by (train_samples, encode_dim_param):")
        print("-" * 70)

        train_samples_list = sorted(summary_df['train_samples'].unique())
        encode_dim_list = sorted(summary_df['encode_dim_param'].unique())

        for train_samples in train_samples_list:
            for encode_dim_param in encode_dim_list:
                subset = summary_df[
                    (summary_df['train_samples'] == train_samples) &
                    (summary_df['encode_dim_param'] == encode_dim_param)
                ]
                if len(subset) == 0:
                    continue

                print(f"\n  samples={int(train_samples)}, dim={int(encode_dim_param)}:")
                for _, row in subset.iterrows():
                    loss_cfg = row.get('loss_config', 'Unknown')
                    fig8_err = row['mean_error_figure8']
                    star_err = row['mean_error_star']
                    stable = row['stability_rate']
                    n = row['n_models']
                    if np.isnan(fig8_err):
                        fig8_str = "N/A"
                    else:
                        fig8_str = f"{fig8_err:.4f}"
                    if np.isnan(star_err):
                        star_str = "N/A"
                    else:
                        star_str = f"{star_err:.4f}"
                    print(f"    {loss_cfg:10s}: fig8={fig8_str}, star={star_str}, stable={stable:.0%} (n={n})")

    # Generate scaling analysis table
    generate_scaling_table(summary_df, output_dir)


def generate_scaling_table(summary_df: pd.DataFrame, output_dir: str):
    """Generate scaling analysis table showing how control performance varies with model size."""
    if 'train_samples' not in summary_df.columns or 'encode_dim_param' not in summary_df.columns:
        return

    print("\n" + "=" * 70)
    print("SCALING ANALYSIS TABLE")
    print("=" * 70)

    loss_configs = ['Baseline', '+Cov', '+Ctrl', '+Aux']

    for loss_cfg in loss_configs:
        cfg_df = summary_df[summary_df['loss_config'] == loss_cfg]
        if len(cfg_df) == 0:
            continue

        print(f"\n{loss_cfg} - Mean Figure-8 Error:")
        print("-" * 50)

        train_samples_list = sorted(cfg_df['train_samples'].unique())
        encode_dim_list = sorted(cfg_df['encode_dim_param'].unique())

        # Header
        header = "samples\\dim |" + " | ".join([f"{int(d):6d}" for d in encode_dim_list])
        print(header)
        print("-" * len(header))

        # Rows
        for train_samples in train_samples_list:
            row_values = []
            for encode_dim_param in encode_dim_list:
                subset = cfg_df[
                    (cfg_df['train_samples'] == train_samples) &
                    (cfg_df['encode_dim_param'] == encode_dim_param)
                ]
                if len(subset) > 0:
                    val = subset['mean_error_figure8'].values[0]
                    if np.isnan(val):
                        row_values.append("   N/A")
                    else:
                        row_values.append(f"{val:6.4f}")
                else:
                    row_values.append("   N/A")
            print(f"{int(train_samples):10d} |" + " | ".join(row_values))

    # Save scaling table as CSV
    scaling_path = os.path.join(output_dir, 'franka_scaling_analysis.csv')
    summary_df.to_csv(scaling_path, index=False)
    print(f"\nSaved scaling analysis to: {scaling_path}")


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="MPC Control Evaluation for Franka Robot"
    )
    parser.add_argument('--project_name', type=str, default='FrankaImproved',
                       help='Project name (log directory)')
    parser.add_argument('--horizon', type=int, default=15,
                       help='MPC horizon')
    parser.add_argument('--warmup_steps', type=int, default=50,
                       help='Warmup steps to skip')
    parser.add_argument('--n_best', type=int, default=1,
                       help='Number of best models per loss config to evaluate')
    parser.add_argument('--largest_only', action='store_true',
                       help='Only evaluate largest model size and sample size (default: evaluate all)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory (default: log/control_results/Franka/{timestamp})')
    parser.add_argument('--log_dir', type=str, default='../log',
                       help='Base log directory')
    # Speed optimization arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu (default: auto)')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Configuration
    config = FrankaMPCConfig(
        horizon=args.horizon,
        warmup_steps=args.warmup_steps,
    )

    # Paths
    log_path = f"../log/{args.project_name}/koopman_results_log.csv"

    # Set up per-environment timestamped output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if args.output_dir is None:
        # Create timestamped output directory: log/control_results/Franka/{timestamp}/
        output_dir = os.path.join(args.log_dir, "control_results", "Franka", timestamp)
    else:
        # Override with user-specified path (backward compatibility)
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("FRANKA MPC CONTROL EVALUATION")
    print("=" * 60)
    print(f"Log path: {log_path}")
    print(f"Output dir: {output_dir}")
    print(f"MPC horizon: {config.horizon}")
    print(f"N best models per config: {args.n_best}")
    print(f"Evaluate all sizes: {not args.largest_only}")
    print(f"Device: {device}")
    print()

    total_start_time = time.time()
    evaluate_all_models(log_path, output_dir, config, n_best=args.n_best,
                       largest_only=args.largest_only, device=device)
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
