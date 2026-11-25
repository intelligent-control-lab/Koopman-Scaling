#!/usr/bin/env python3
"""
Baseline MPC Control for Franka Robot
======================================

This script implements MPC control evaluation for baseline models (MLP and EDMD)
on Franka robot end-effector trajectory tracking tasks.

Trajectories:
- Figure-8 (Lemniscate of Bernoulli)
- 5-point Star

Supports two baseline methods:
1. MLP + Random Shooting MPC (nonlinear dynamics)
2. EDMD + Linear MPC (polynomial lifting with analytical QP)

Key Features:
- Same evaluation metrics as Koopman MPC (tracking error, max error)
- Uses same PyBullet simulation as franka_mpc.py
- Outputs comparable CSV format for paper tables

Usage:
    python baseline_franka_mpc.py --method MLP
    python baseline_franka_mpc.py --method EDMD
    python baseline_franka_mpc.py --method all

Author: Koopman Scaling Law Analysis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.linalg
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utility'))

from baseline_mpc_utils import (
    load_mlp_from_checkpoint,
    load_edmd_from_checkpoint,
)
from mlp_mpc import MLPShootingMPC
from edmd_mpc import EDMD_MPC

# Try to import PyBullet
try:
    import pybullet as pb
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utility'))
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

    # Cost weights
    Q_position: float = 10.0
    Q_joint: float = 1.0

    # Evaluation
    warmup_steps: int = 100
    noise_sigma: float = 0.01

    # Action clipping
    action_low: float = -0.3
    action_high: float = 0.3

    # Stability
    stability_threshold: float = 1.10

    # MLP Shooting MPC
    num_samples: int = 1000


# Franka environment info
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
    """
    q = full_state[7:14]   # joint positions
    dq = full_state[14:21]  # joint velocities
    return np.concatenate([q, dq])


def get_ee_position(env) -> np.ndarray:
    """Get end-effector xyz position from environment."""
    ee_state = pb.getLinkState(env.robot, env.ee_id)
    return np.array(ee_state[4])


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
# Control Evaluation
# =============================================================================
@dataclass
class FrankaControlResult:
    """Results from Franka control evaluation."""
    error_figure8: float = float('inf')
    error_star: float = float('inf')
    max_error_figure8: float = float('inf')
    max_error_star: float = float('inf')
    method: str = ''
    stable: bool = False
    eval_time_s: float = 0.0


def evaluate_mlp_franka_mpc(model_path: str,
                            config: FrankaMPCConfig = None,
                            device: str = 'cpu') -> FrankaControlResult:
    """
    Evaluate MLP + Random Shooting MPC for Franka trajectory tracking.

    Args:
        model_path: Path to trained MLP model
        config: MPC configuration
        device: torch device

    Returns:
        FrankaControlResult with tracking errors
    """
    if not PYBULLET_AVAILABLE:
        return FrankaControlResult(method='MLP')

    if config is None:
        config = FrankaMPCConfig()

    start_time = time.time()

    state_dim = FRANKA_CONFIG['state_dim']
    action_dim = FRANKA_CONFIG['action_dim']
    dt = FRANKA_CONFIG['dt']

    # Load model
    try:
        model, norm_stats = load_mlp_from_checkpoint(model_path, device=device)
    except Exception as e:
        print(f"  Error loading MLP model: {e}")
        return FrankaControlResult(method='MLP')

    # Create MPC controller
    mpc = MLPShootingMPC(
        model=model,
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=config.horizon,
        num_samples=config.num_samples,
        action_low=config.action_low,
        action_high=config.action_high,
        Q_state=config.Q_position,
        Q_vel=config.Q_joint,
        angle_idx=list(range(7)),    # Joint positions
        vel_idx=list(range(7, 14)),  # Joint velocities
        norm_stats=norm_stats,
        device=device,
    )

    # Create environment
    try:
        env = FrankaEnv(render=False)
    except Exception as e:
        print(f"  Error creating Franka env: {e}")
        return FrankaControlResult(method='MLP')

    def run_trajectory(traj_positions: np.ndarray, traj_name: str) -> Tuple[float, float]:
        """Run a trajectory and compute tracking error."""
        T = len(traj_positions)
        env.reset()

        # Compute joint positions via IK for full reference trajectory
        ref_states = np.zeros((T, state_dim))
        for i in range(T):
            joints = ik_to(env, traj_positions[i, 0], traj_positions[i, 1], traj_positions[i, 2])
            ref_states[i, :7] = joints

        # Initialize at first reference (slightly offset)
        joints_init = ik_to(env, traj_positions[0, 0],
                           traj_positions[0, 1] + 0.01,
                           traj_positions[0, 2] + 0.043)
        for j, jnt in enumerate(joints_init):
            pb.resetJointState(env.robot, j, float(jnt))

        # Get initial state
        full_state = env.get_state()
        errors = []

        for step in range(T - config.horizon - 1):
            # Extract joint-only state
            joint_state = extract_joint_state(full_state)

            # Add observation noise
            noise = np.random.randn(7) * config.noise_sigma
            joint_state_noisy = joint_state.copy()
            joint_state_noisy[:7] += noise

            # Reference window
            ref_window = ref_states[step:step+config.horizon+1]

            # Get MPC control (use trajectory solving)
            try:
                u = mpc.solve_trajectory(joint_state_noisy, ref_window)
            except Exception as e:
                print(f"  MPC solve error at step {step}: {e}")
                break

            # Clip and apply
            u = np.clip(u, config.action_low, config.action_high)
            full_state = env.step(u)

            if step >= config.warmup_steps:
                ee_pos = get_ee_position(env)
                err = np.linalg.norm(ee_pos[1:3] - traj_positions[step, 1:3])
                errors.append(err)

        if len(errors) > 0:
            return np.mean(errors), np.max(errors)
        return float('inf'), float('inf')

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

    eval_time = time.time() - start_time

    return FrankaControlResult(
        error_figure8=error_fig8,
        error_star=error_star,
        max_error_figure8=max_error_fig8,
        max_error_star=max_error_star,
        method='MLP',
        stable=True,
        eval_time_s=eval_time,
    )


def evaluate_edmd_franka_mpc(model_path: str,
                             config: FrankaMPCConfig = None) -> FrankaControlResult:
    """
    Evaluate EDMD + Linear MPC for Franka trajectory tracking.

    Args:
        model_path: Path to trained EDMD model
        config: MPC configuration

    Returns:
        FrankaControlResult with tracking errors
    """
    if not PYBULLET_AVAILABLE:
        return FrankaControlResult(method='EDMD')

    if config is None:
        config = FrankaMPCConfig()

    start_time = time.time()

    state_dim = FRANKA_CONFIG['state_dim']
    action_dim = FRANKA_CONFIG['action_dim']
    dt = FRANKA_CONFIG['dt']

    # Load model
    try:
        model, norm_stats = load_edmd_from_checkpoint(model_path)
    except Exception as e:
        print(f"  Error loading EDMD model: {e}")
        return FrankaControlResult(method='EDMD')

    # Check stability
    rho_A = spectral_radius(model.A)
    if rho_A > config.stability_threshold:
        print(f"  Unstable EDMD dynamics: rho(A) = {rho_A:.4f}")
        return FrankaControlResult(method='EDMD')

    # Create MPC controller
    mpc = EDMD_MPC(
        edmd_model=model,
        horizon=config.horizon,
        Q_state=config.Q_position,
        Q_vel=config.Q_joint,
        action_low=config.action_low,
        action_high=config.action_high,
        angle_idx=list(range(7)),
        vel_idx=list(range(7, 14)),
        norm_stats=norm_stats,
    )

    # Create environment
    try:
        env = FrankaEnv(render=False)
    except Exception as e:
        print(f"  Error creating Franka env: {e}")
        return FrankaControlResult(method='EDMD')

    def run_trajectory(traj_positions: np.ndarray, traj_name: str) -> Tuple[float, float]:
        """Run a trajectory and compute tracking error."""
        T = len(traj_positions)
        env.reset()

        # Compute joint positions via IK for full reference trajectory
        ref_states = np.zeros((T, state_dim))
        for i in range(T):
            joints = ik_to(env, traj_positions[i, 0], traj_positions[i, 1], traj_positions[i, 2])
            ref_states[i, :7] = joints

        # Initialize at first reference (slightly offset)
        joints_init = ik_to(env, traj_positions[0, 0],
                           traj_positions[0, 1] + 0.01,
                           traj_positions[0, 2] + 0.043)
        for j, jnt in enumerate(joints_init):
            pb.resetJointState(env.robot, j, float(jnt))

        # Get initial state
        full_state = env.get_state()
        errors = []

        for step in range(T - config.horizon - 1):
            # Extract joint-only state
            joint_state = extract_joint_state(full_state)

            # Add observation noise
            noise = np.random.randn(7) * config.noise_sigma
            joint_state_noisy = joint_state.copy()
            joint_state_noisy[:7] += noise

            # Reference window
            ref_window = ref_states[step:step+config.horizon+1]

            # Get MPC control
            try:
                u = mpc.solve_trajectory(joint_state_noisy, ref_window)
            except Exception as e:
                print(f"  MPC solve error at step {step}: {e}")
                break

            # Clip and apply
            u = np.clip(u, config.action_low, config.action_high)
            full_state = env.step(u)

            if step >= config.warmup_steps:
                ee_pos = get_ee_position(env)
                err = np.linalg.norm(ee_pos[1:3] - traj_positions[step, 1:3])
                errors.append(err)

        if len(errors) > 0:
            return np.mean(errors), np.max(errors)
        return float('inf'), float('inf')

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

    eval_time = time.time() - start_time

    return FrankaControlResult(
        error_figure8=error_fig8,
        error_star=error_star,
        max_error_figure8=max_error_fig8,
        max_error_star=max_error_star,
        method='EDMD',
        stable=True,
        eval_time_s=eval_time,
    )


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================
def find_baseline_models(log_dir: str, env_name: str, method: str) -> List[Dict]:
    """
    Find available baseline models for evaluation.

    Args:
        log_dir: Base log directory
        env_name: Environment name (e.g., 'FrankaImproved')
        method: 'MLP' or 'EDMD'

    Returns:
        List of model info dicts
    """
    best_models_dir = os.path.join(log_dir, 'best_models')
    if not os.path.exists(best_models_dir):
        print(f"  Best models directory not found: {best_models_dir}")
        return []

    models = []
    prefix = method.lower() + '_' + env_name

    for fname in os.listdir(best_models_dir):
        if fname.startswith(prefix) and fname.endswith('.pt'):
            parts = fname[:-3].split('_')
            try:
                train_samples = int(parts[-1])
                encode_dim_param = int(parts[-2])
                seed = int(parts[-3])

                models.append({
                    'model_path': os.path.join(best_models_dir, fname),
                    'env_name': env_name,
                    'method': method,
                    'seed': seed,
                    'encode_dim_param': encode_dim_param,
                    'train_samples': train_samples,
                })
            except (ValueError, IndexError):
                continue

    return models


def run_evaluation(method: str, log_dir: str, env_name: str = 'FrankaImproved',
                   config: FrankaMPCConfig = None,
                   device: str = 'cpu') -> pd.DataFrame:
    """
    Run evaluation for all Franka models matching method.

    Args:
        method: 'MLP' or 'EDMD'
        log_dir: Base log directory
        env_name: Environment name
        config: MPC configuration
        device: torch device

    Returns:
        DataFrame with results
    """
    if config is None:
        config = FrankaMPCConfig()

    # Find available models
    models = find_baseline_models(log_dir, env_name, method)
    if not models:
        print(f"No {method} models found for {env_name}")
        return pd.DataFrame()

    print(f"Found {len(models)} {method} models for {env_name}")

    results = []
    for model_info in tqdm(models, desc=f"Evaluating {method} on Franka"):
        model_path = model_info['model_path']
        print(f"\nEvaluating: {os.path.basename(model_path)}")

        # Run evaluation
        if method == 'MLP':
            result = evaluate_mlp_franka_mpc(model_path, config, device)
        else:  # EDMD
            result = evaluate_edmd_franka_mpc(model_path, config)

        # Combine model info with result
        row = {
            'env_name': env_name,
            'method': method,
            'model_path': model_path,
            'seed': model_info['seed'],
            'encode_dim_param': model_info['encode_dim_param'],
            'train_samples': model_info['train_samples'],
            'error_figure8': result.error_figure8,
            'error_star': result.error_star,
            'max_error_figure8': result.max_error_figure8,
            'max_error_star': result.max_error_star,
            'stable': result.stable,
            'eval_time_s': result.eval_time_s,
        }
        results.append(row)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description='Baseline MPC Control for Franka')
    parser.add_argument('--method', type=str, default='all',
                        choices=['MLP', 'EDMD', 'all'],
                        help='Baseline method to evaluate')
    parser.add_argument('--env_name', type=str, default='FrankaImproved',
                        help='Environment name')
    parser.add_argument('--log_dir', type=str,
                        default='../../../log/BaselineResults_Full_seed2',
                        help='Base log directory with baseline models')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: log_dir)')
    parser.add_argument('--horizon', type=int, default=15, help='MPC horizon')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for MLP shooting MPC')
    parser.add_argument('--device', type=str, default='cpu', help='torch device')

    args = parser.parse_args()

    if not PYBULLET_AVAILABLE:
        print("PyBullet is required for Franka evaluation. Please install with: pip install pybullet")
        return

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, args.log_dir)
    output_dir = args.output_dir or log_dir

    if not os.path.exists(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Create config
    config = FrankaMPCConfig(
        horizon=args.horizon,
        num_samples=args.num_samples,
    )

    # Determine methods to evaluate
    methods = ['MLP', 'EDMD'] if args.method == 'all' else [args.method]

    all_results = []

    for method in methods:
        print(f"\n{'='*60}")
        print(f"Evaluating {method} on Franka")
        print(f"{'='*60}")

        df = run_evaluation(method, log_dir, args.env_name, config, args.device)

        if len(df) > 0:
            all_results.append(df)

            print(f"\nSummary for {method}:")
            print(f"  Models evaluated: {len(df)}")
            print(f"  Mean Figure-8 error: {df['error_figure8'].mean():.4f}")
            print(f"  Mean Star error: {df['error_star'].mean():.4f}")

    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(output_dir, 'baseline_franka_mpc_results.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        # Print final summary
        print(f"\n{'='*60}")
        print("Final Summary")
        print(f"{'='*60}")
        summary = combined_df.groupby('method').agg({
            'error_figure8': 'mean',
            'error_star': 'mean',
            'stable': 'sum',
        }).round(4)
        print(summary)


if __name__ == '__main__':
    main()