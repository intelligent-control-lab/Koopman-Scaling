#!/usr/bin/env python3
"""
MLP + Random Shooting MPC
=========================

Model Predictive Control using MLP dynamics with Random Shooting optimization.

Since MLP is a nonlinear model, we cannot use analytical QP solutions.
Instead, we use random shooting: sample many control sequences, rollout
each using the MLP, and select the one with minimum cost.

Author: Koopman Scaling Law Analysis
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

try:
    from .baseline_mpc_utils import (
        MLPDynamics,
        normalize_state,
        denormalize_action,
    )
except ImportError:
    from baseline_mpc_utils import (
        MLPDynamics,
        normalize_state,
        denormalize_action,
    )


@dataclass
class MLPMPCConfig:
    """Configuration for MLP Shooting MPC."""
    # MPC settings
    horizon: int = 15
    num_samples: int = 1000

    # Cost weights
    Q_state: float = 10.0
    Q_vel: float = 1.0

    # Action bounds (will be overridden by environment-specific settings)
    action_low: float = -1.0
    action_high: float = 1.0


class MLPShootingMPC:
    """
    MPC using MLP dynamics with Random Shooting optimization.

    Solves:
        min  sum_{k=0}^{H} ||x_k - x_ref||_Q^2

    Using random sampling:
    1. Sample num_samples random control sequences uniformly within action bounds
    2. Rollout each sequence using MLP autoregressive prediction
    3. Compute cost for each trajectory (weighted state error)
    4. Select sequence with minimum cost
    5. Return first control from best sequence
    """

    def __init__(self, model: MLPDynamics, state_dim: int, action_dim: int,
                 horizon: int = 15, num_samples: int = 1000,
                 action_low: float = -1.0, action_high: float = 1.0,
                 Q_state: float = 10.0, Q_vel: float = 1.0,
                 angle_idx: list = None, vel_idx: list = None,
                 norm_stats: Optional[Dict] = None,
                 device: str = 'cpu'):
        """
        Initialize MLP Shooting MPC.

        Args:
            model: MLPDynamics model
            state_dim: State dimension
            action_dim: Action dimension
            horizon: MPC horizon
            num_samples: Number of random control sequences to sample
            action_low: Lower action bound
            action_high: Upper action bound
            Q_state: Cost weight on position/angle states
            Q_vel: Cost weight on velocity states
            angle_idx: Indices of angle states (for higher weight)
            vel_idx: Indices of velocity states (for lower weight)
            norm_stats: Normalization statistics
            device: torch device
        """
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.H = horizon
        self.num_samples = num_samples
        self.action_low = action_low
        self.action_high = action_high
        self.norm_stats = norm_stats
        self.device = device

        # Build Q weights
        self.Q_weights = np.ones(state_dim)
        if angle_idx is not None:
            for i in angle_idx:
                self.Q_weights[i] = Q_state
        if vel_idx is not None:
            for i in vel_idx:
                self.Q_weights[i] = Q_vel

        # If no indices provided, use default
        if angle_idx is None and vel_idx is None:
            # Default: first half are positions, second half are velocities
            half = state_dim // 2
            self.Q_weights[:half] = Q_state
            self.Q_weights[half:] = Q_vel

    def _normalize_state_batch(self, states: np.ndarray) -> np.ndarray:
        """Normalize a batch of states."""
        if self.norm_stats is None:
            return states
        state_mean = np.array(self.norm_stats['state_mean'])
        state_std = np.array(self.norm_stats['state_std'])
        state_std = np.maximum(state_std, 1e-8)
        return (states - state_mean) / state_std

    def rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """
        Rollout MLP dynamics for multiple control sequences.

        Args:
            x0: Initial state [state_dim]
            u_seq: Control sequences [num_samples, horizon, action_dim]

        Returns:
            x_pred: Predicted trajectories [num_samples, horizon, state_dim]
        """
        num_samples, horizon, _ = u_seq.shape

        with torch.inference_mode():
            # Normalize initial state
            x0_norm = normalize_state(x0, self.norm_stats)

            # Expand x0 for batch
            x0_batch = np.tile(x0_norm, (num_samples, 1))

            # Convert to tensors
            x_tensor = torch.tensor(x0_batch, dtype=torch.float32, device=self.device)
            u_tensor = torch.tensor(u_seq, dtype=torch.float32, device=self.device)

            # Transpose u_seq to [horizon, num_samples, action_dim]
            u_tensor = u_tensor.permute(1, 0, 2)

            # Multi-step prediction
            x_pred = self.model.predict_ksteps(x_tensor, u_tensor)

            # Transpose to [num_samples, horizon, state_dim]
            x_pred = x_pred.permute(1, 0, 2).cpu().numpy()

        return x_pred

    def compute_cost(self, x_pred: np.ndarray, target_state: np.ndarray) -> np.ndarray:
        """
        Compute cost for each trajectory.

        Args:
            x_pred: Predicted trajectories [num_samples, horizon, state_dim]
            target_state: Target state [state_dim]

        Returns:
            costs: Cost for each trajectory [num_samples]
        """
        # Normalize target
        target_norm = normalize_state(target_state, self.norm_stats)

        # Compute weighted squared error
        errors = x_pred - target_norm  # [num_samples, horizon, state_dim]
        weighted_errors = errors ** 2 * self.Q_weights  # [num_samples, horizon, state_dim]

        # Sum over horizon and state dimensions
        costs = np.sum(weighted_errors, axis=(1, 2))  # [num_samples]

        return costs

    def compute_trajectory_cost(self, x_pred: np.ndarray,
                                 ref_trajectory: np.ndarray) -> np.ndarray:
        """
        Compute cost for trajectory tracking.

        Args:
            x_pred: Predicted trajectories [num_samples, horizon, state_dim]
            ref_trajectory: Reference trajectory [horizon, state_dim]

        Returns:
            costs: Cost for each trajectory [num_samples]
        """
        num_samples, horizon, state_dim = x_pred.shape

        # Normalize reference trajectory
        ref_norm = self._normalize_state_batch(ref_trajectory)

        # Compute weighted squared error
        errors = x_pred - ref_norm  # [num_samples, horizon, state_dim]
        weighted_errors = errors ** 2 * self.Q_weights  # [num_samples, horizon, state_dim]

        # Sum over horizon and state dimensions
        costs = np.sum(weighted_errors, axis=(1, 2))  # [num_samples]

        return costs

    def solve(self, current_state: np.ndarray,
              target_state: np.ndarray) -> np.ndarray:
        """
        Solve MPC using random shooting.

        Args:
            current_state: Current physical state [state_dim]
            target_state: Target physical state [state_dim]

        Returns:
            u: Optimal control input in physical units [action_dim]
        """
        # Sample random control sequences uniformly
        u_samples = np.random.uniform(
            self.action_low, self.action_high,
            size=(self.num_samples, self.H, self.action_dim)
        )

        # If using normalization, sample in normalized space
        if self.norm_stats is not None:
            # Sample in normalized action space (typically ~[-1, 1])
            action_mean = np.array(self.norm_stats.get('action_mean', 0))
            action_std = np.array(self.norm_stats.get('action_std', 1))
            # Convert bounds to normalized space
            norm_low = (self.action_low - action_mean) / np.maximum(action_std, 1e-8)
            norm_high = (self.action_high - action_mean) / np.maximum(action_std, 1e-8)
            u_samples = np.random.uniform(
                norm_low, norm_high,
                size=(self.num_samples, self.H, self.action_dim)
            )

        # Rollout all sequences
        x_pred = self.rollout(current_state, u_samples)

        # Compute costs
        costs = self.compute_cost(x_pred, target_state)

        # Select best sequence
        best_idx = np.argmin(costs)
        u_best = u_samples[best_idx, 0]  # First control from best sequence

        # Denormalize action
        u = denormalize_action(u_best, self.norm_stats)

        # Clip to bounds
        u = np.clip(u, self.action_low, self.action_high)

        return u

    def solve_trajectory(self, current_state: np.ndarray,
                         ref_trajectory: np.ndarray) -> np.ndarray:
        """
        Solve MPC with trajectory reference.

        Args:
            current_state: Current state in physical units [state_dim]
            ref_trajectory: Reference states over horizon [H, state_dim] or [H+1, state_dim]

        Returns:
            u: Optimal control input in physical units [action_dim]
        """
        # Ensure ref_trajectory has correct length
        if len(ref_trajectory) > self.H:
            ref_trajectory = ref_trajectory[:self.H]
        elif len(ref_trajectory) < self.H:
            # Pad with last reference
            padding = np.tile(ref_trajectory[-1], (self.H - len(ref_trajectory), 1))
            ref_trajectory = np.vstack([ref_trajectory, padding])

        # Sample random control sequences
        if self.norm_stats is not None:
            action_mean = np.array(self.norm_stats.get('action_mean', 0))
            action_std = np.array(self.norm_stats.get('action_std', 1))
            norm_low = (self.action_low - action_mean) / np.maximum(action_std, 1e-8)
            norm_high = (self.action_high - action_mean) / np.maximum(action_std, 1e-8)
            u_samples = np.random.uniform(
                norm_low, norm_high,
                size=(self.num_samples, self.H, self.action_dim)
            )
        else:
            u_samples = np.random.uniform(
                self.action_low, self.action_high,
                size=(self.num_samples, self.H, self.action_dim)
            )

        # Rollout all sequences
        x_pred = self.rollout(current_state, u_samples)

        # Compute costs
        costs = self.compute_trajectory_cost(x_pred, ref_trajectory)

        # Select best sequence
        best_idx = np.argmin(costs)
        u_best = u_samples[best_idx, 0]

        # Denormalize action
        u = denormalize_action(u_best, self.norm_stats)

        # Clip to bounds
        u = np.clip(u, self.action_low, self.action_high)

        return u