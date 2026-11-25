#!/usr/bin/env python3
"""
EDMD + Linear MPC
=================

Model Predictive Control using EDMD dynamics with analytical QP solution.

Since EDMD provides linear dynamics in the lifted polynomial space:
    z_{t+1} = A z_t + B u_t

We can use the same analytical QP solution as Koopman MPC.

Author: Koopman Scaling Law Analysis
"""

import numpy as np
import scipy.linalg
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

try:
    from .baseline_mpc_utils import (
        EDMD,
        normalize_state,
        denormalize_action,
    )
except ImportError:
    from baseline_mpc_utils import (
        EDMD,
        normalize_state,
        denormalize_action,
    )


@dataclass
class EDMD_MPCConfig:
    """Configuration for EDMD Linear MPC."""
    # MPC settings
    horizon: int = 15

    # Cost weights
    Q_state: float = 10.0
    Q_vel: float = 1.0

    # Action bounds
    action_low: float = -1.0
    action_high: float = 1.0

    # Stability
    stability_threshold: float = 1.05


class EDMD_MPC:
    """
    MPC using EDMD linear dynamics with analytical QP solution.

    Solves:
        min  sum_{k=0}^{H} ||z_k - z_ref||_Q^2 + sum_{k=0}^{H-1} ||u_k||_R^2
        s.t. z_{k+1} = A z_k + B u_k

    Using analytical solution for unconstrained QP (same as Koopman MPC).
    """

    def __init__(self, edmd_model: EDMD, horizon: int = 15,
                 Q_state: float = 10.0, Q_vel: float = 1.0,
                 action_low: float = -1.0, action_high: float = 1.0,
                 angle_idx: list = None, vel_idx: list = None,
                 norm_stats: Optional[Dict] = None):
        """
        Initialize EDMD Linear MPC.

        Args:
            edmd_model: EDMD model with A, B matrices
            horizon: MPC horizon
            Q_state: Cost weight on position/angle states
            Q_vel: Cost weight on velocity states
            action_low: Lower action bound
            action_high: Upper action bound
            angle_idx: Indices of angle states
            vel_idx: Indices of velocity states
            norm_stats: Normalization statistics
        """
        self.edmd = edmd_model
        self.A = np.asarray(edmd_model.A, dtype=np.float64)
        self.B = np.asarray(edmd_model.B, dtype=np.float64)
        self.H = horizon
        self.action_low = action_low
        self.action_high = action_high
        self.norm_stats = norm_stats

        self.lift_dim = edmd_model.lift_dim
        self.state_dim = edmd_model.state_dim
        self.action_dim = edmd_model.u_dim

        self.angle_idx = angle_idx
        self.vel_idx = vel_idx
        self.Q_state = Q_state
        self.Q_vel = Q_vel

        # Build cost matrices
        self.Q = self._build_Q()
        self.R = np.zeros((self.action_dim, self.action_dim))  # R = 0 (match Koopman)

        # Precompute MPC matrices
        self._precompute_matrices()

    def _build_Q(self) -> np.ndarray:
        """
        Build state cost matrix.

        Cost only on original state dimensions (first state_dim).
        Zero cost on polynomial features (matching Koopman latent).
        """
        Q = np.zeros((self.lift_dim, self.lift_dim))

        # Build weights for original state dimensions
        if self.angle_idx is not None:
            for i in self.angle_idx:
                Q[i, i] = self.Q_state
        if self.vel_idx is not None:
            for i in self.vel_idx:
                Q[i, i] = self.Q_vel

        # If no indices provided, use default
        if self.angle_idx is None and self.vel_idx is None:
            half = self.state_dim // 2
            for i in range(half):
                Q[i, i] = self.Q_state
            for i in range(half, self.state_dim):
                Q[i, i] = self.Q_vel

        # Polynomial features: ZERO cost (matching Koopman latent states)
        # Q[self.state_dim:, self.state_dim:] = 0.0  (already zero)

        return Q

    def _precompute_matrices(self):
        """Precompute block matrices for efficient MPC solve."""
        H = self.H
        n = self.lift_dim
        m = self.action_dim

        # State propagation matrix M: [I; A; A^2; ...; A^H]
        self.M = np.vstack([np.linalg.matrix_power(self.A, i) for i in range(H + 1)])

        # Control influence matrix C
        self.C = np.zeros(((H + 1) * n, H * m))
        for i in range(1, H + 1):
            for j in range(i):
                self.C[i*n:(i+1)*n, j*m:(j+1)*m] = (
                    np.linalg.matrix_power(self.A, i-j-1) @ self.B
                )

        # Terminal cost F (same as Q)
        F = self.Q.copy()

        # Block diagonal cost matrices
        Qh = scipy.linalg.block_diag(*([self.Q] * H), F)
        Rh = scipy.linalg.block_diag(*([self.R] * H))

        # QP matrices
        self.P = 2 * (Rh + self.C.T @ Qh @ self.C)
        self.P_inv = np.linalg.inv(0.5 * self.P)
        self.Qh = Qh

    def encode(self, state: np.ndarray) -> np.ndarray:
        """
        Encode state to lifted polynomial space.

        If normalization stats are available, normalizes the state first.
        """
        # Normalize state if norm_stats available
        state_norm = normalize_state(state, self.norm_stats)

        # Lift to polynomial space
        z = self.edmd.lift(state_norm.reshape(1, -1)).flatten()
        return z

    def spectral_radius(self) -> float:
        """Compute spectral radius of A matrix."""
        return float(np.max(np.abs(scipy.linalg.eigvals(self.A))))

    def solve(self, current_state: np.ndarray,
              target_state: np.ndarray) -> np.ndarray:
        """
        Solve MPC to get optimal control input.

        Args:
            current_state: Current physical state [state_dim]
            target_state: Target physical state [state_dim]

        Returns:
            u: Optimal control input in physical units [action_dim]
        """
        # Encode current and target states
        z_current = self.encode(current_state)
        z_target = self.encode(target_state)

        # Reference trajectory: constant target over horizon
        Z_ref = np.tile(z_target, self.H + 1)

        # QP gradient: q = 2 * (M @ z_current - Z_ref)^T @ Qh @ C
        predicted = self.M @ z_current
        q = 2 * (predicted - Z_ref) @ self.Qh @ self.C

        # Analytical solution: U = -0.5 * P^-1 @ q^T
        U_opt = -0.5 * self.P_inv @ q.T

        # Extract first control input (in normalized space)
        u_norm = U_opt[:self.action_dim]

        # Denormalize action to physical units
        u = denormalize_action(u_norm, self.norm_stats)

        # Clip to bounds
        u = np.clip(u, self.action_low, self.action_high)

        return u

    def solve_trajectory(self, current_state: np.ndarray,
                         ref_trajectory: np.ndarray) -> np.ndarray:
        """
        Solve MPC with trajectory reference.

        Args:
            current_state: Current state in physical units [state_dim]
            ref_trajectory: Reference states over horizon [H+1, state_dim]

        Returns:
            u: Optimal control input in physical units [action_dim]
        """
        # Encode current state
        z_current = self.encode(current_state)

        # Encode reference trajectory
        Z_ref = np.zeros((self.H + 1) * self.lift_dim)
        for i in range(min(len(ref_trajectory), self.H + 1)):
            z_ref_i = self.encode(ref_trajectory[i])
            Z_ref[i*self.lift_dim:(i+1)*self.lift_dim] = z_ref_i

        # Fill remaining with last reference
        if len(ref_trajectory) < self.H + 1:
            z_last = self.encode(ref_trajectory[-1])
            for i in range(len(ref_trajectory), self.H + 1):
                Z_ref[i*self.lift_dim:(i+1)*self.lift_dim] = z_last

        # QP solution
        predicted = self.M @ z_current
        q = 2 * (predicted - Z_ref) @ self.Qh @ self.C
        U_opt = -0.5 * self.P_inv @ q.T

        # Extract first control
        u_norm = U_opt[:self.action_dim]

        # Denormalize action to physical units
        u = denormalize_action(u_norm, self.norm_stats)

        # Clip to bounds
        u = np.clip(u, self.action_low, self.action_high)

        return u