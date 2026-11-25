#!/usr/bin/env python3
"""
Baseline MPC Utilities
======================

Shared utilities for baseline dynamics models (MLP and EDMD) in MPC control.

Contains:
- MLPDynamics: Standard MLP that predicts x_{t+1} from (x_t, u_t)
- EDMD: Extended Dynamic Mode Decomposition with polynomial lifting
- Model loading functions
- Normalization utilities

Author: Koopman Scaling Law Analysis
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, List


# ============================================================================
# MLP Dynamics Model
# ============================================================================

class MLPDynamics(nn.Module):
    """
    Standard MLP that predicts x_{t+1} from (x_t, u_t).

    Architecture: Input -> [Linear -> ReLU] * hidden_layers -> Linear -> Output
    """
    def __init__(self, state_dim: int, u_dim: int, hidden_dim: int = 256,
                 hidden_layers: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.u_dim = u_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers

        input_dim = state_dim + u_dim
        output_dim = state_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Predict next state from current state and control."""
        xu = torch.cat([x, u], dim=-1)
        return self.net(xu)

    def predict_ksteps(self, x0: torch.Tensor, u_seq: torch.Tensor) -> torch.Tensor:
        """
        Multi-step autoregressive prediction.

        Args:
            x0: Initial state [batch, state_dim]
            u_seq: Control sequence [K, batch, u_dim]

        Returns:
            x_pred: Predicted states [K, batch, state_dim]
        """
        K, batch, _ = u_seq.shape
        x_pred = torch.zeros(K, batch, self.state_dim, device=x0.device, dtype=x0.dtype)

        x = x0
        for k in range(K):
            x = self.forward(x, u_seq[k])
            x_pred[k] = x

        return x_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# EDMD with Hierarchical Polynomial Lifting
# ============================================================================

class EDMD:
    """
    Extended Dynamic Mode Decomposition with hierarchical polynomial lifting.

    Model: z_{t+1} = A @ z_t + B @ u_t
    where z = lift(x) includes polynomial features.

    Hierarchical polynomial lifting (matching Koopman dimension exactly):
    Features are added in this order until reaching target dimension:
    1. 1st order (linear): [x_1, x_2, ..., x_n]
    2. 2nd order powers: [x_1^2, x_2^2, ..., x_n^2]
    3. 3rd order powers: [x_1^3, x_2^3, ..., x_n^3]
    4. 2nd order cross terms: [x_1*x_2, x_1*x_3, ..., x_{n-1}*x_n]
    5. 3rd order cross terms if needed
    6. 4th order powers and cross terms if still below target

    Target dimension: state_dim * (1 + encode_dim_param) to match Koopman.
    """
    def __init__(self, state_dim: int, u_dim: int, encode_dim_param: int = 1):
        self.state_dim = state_dim
        self.u_dim = u_dim
        self.encode_dim_param = encode_dim_param

        # Target dimension to match Koopman: Nkoopman = state_dim * (1 + encode_dim_param)
        self.target_dim = state_dim * (1 + encode_dim_param)

        # Build feature specification hierarchically
        self.feature_specs = self._build_feature_specs()
        self.lift_dim = len(self.feature_specs)

        # For numerical stability, cap the lift_dim if it would cause issues
        max_safe_dim = min(self.target_dim, 5000)
        if self.lift_dim > max_safe_dim:
            self.feature_specs = self.feature_specs[:max_safe_dim]
            self.lift_dim = max_safe_dim

        self.A = None  # [lift_dim, lift_dim]
        self.B = None  # [lift_dim, u_dim]

        # For saving/loading
        self.fitted = False

        # Store max degree used for info
        self.max_degree = max(sum(spec) for spec in self.feature_specs) if self.feature_specs else 1

    def _build_feature_specs(self) -> list:
        """
        Build hierarchical feature specifications.

        Each spec is a tuple of exponents for each state variable.
        E.g., (1, 0, 0) means x_1^1 * x_2^0 * x_3^0 = x_1
        """
        n = self.state_dim
        target = self.target_dim
        specs = []

        # 1. First order (linear terms): [x_1, x_2, ..., x_n]
        for i in range(n):
            exp = [0] * n
            exp[i] = 1
            specs.append(tuple(exp))
            if len(specs) >= target:
                return specs

        # 2. Second order powers: [x_1^2, x_2^2, ..., x_n^2]
        for i in range(n):
            exp = [0] * n
            exp[i] = 2
            specs.append(tuple(exp))
            if len(specs) >= target:
                return specs

        # 3. Third order powers: [x_1^3, x_2^3, ..., x_n^3]
        for i in range(n):
            exp = [0] * n
            exp[i] = 3
            specs.append(tuple(exp))
            if len(specs) >= target:
                return specs

        # 4. Second order cross terms: [x_i * x_j for i < j]
        for i in range(n):
            for j in range(i + 1, n):
                exp = [0] * n
                exp[i] = 1
                exp[j] = 1
                specs.append(tuple(exp))
                if len(specs) >= target:
                    return specs

        # 5. Third order cross terms: [x_i^2 * x_j, x_i * x_j^2, x_i * x_j * x_k]
        for i in range(n):
            for j in range(n):
                if i != j:
                    exp = [0] * n
                    exp[i] = 2
                    exp[j] = 1
                    specs.append(tuple(exp))
                    if len(specs) >= target:
                        return specs

        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    exp = [0] * n
                    exp[i] = 1
                    exp[j] = 1
                    exp[k] = 1
                    specs.append(tuple(exp))
                    if len(specs) >= target:
                        return specs

        # 6. Fourth order powers: [x_1^4, x_2^4, ..., x_n^4]
        for i in range(n):
            exp = [0] * n
            exp[i] = 4
            specs.append(tuple(exp))
            if len(specs) >= target:
                return specs

        # 7. Fourth order cross terms
        for i in range(n):
            for j in range(n):
                if i != j:
                    exp = [0] * n
                    exp[i] = 3
                    exp[j] = 1
                    specs.append(tuple(exp))
                    if len(specs) >= target:
                        return specs

        for i in range(n):
            for j in range(i + 1, n):
                exp = [0] * n
                exp[i] = 2
                exp[j] = 2
                specs.append(tuple(exp))
                if len(specs) >= target:
                    return specs

        # Fifth order powers
        for i in range(n):
            exp = [0] * n
            exp[i] = 5
            specs.append(tuple(exp))
            if len(specs) >= target:
                return specs

        return specs

    def lift(self, x: np.ndarray) -> np.ndarray:
        """
        Lift state to hierarchical polynomial feature space.

        Args:
            x: State array [batch, state_dim] or [state_dim]

        Returns:
            z: Lifted state [batch, lift_dim]
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        batch = x.shape[0]
        z = np.zeros((batch, self.lift_dim), dtype=np.float64)

        # Clip extreme values for numerical stability
        x_safe = np.clip(x, -10.0, 10.0)

        # Compute each polynomial feature
        for feat_idx, exponents in enumerate(self.feature_specs):
            feat = np.ones(batch, dtype=np.float64)
            for var_idx, exp in enumerate(exponents):
                if exp > 0:
                    feat *= np.power(x_safe[:, var_idx], exp)

            z[:, feat_idx] = np.clip(feat, -1e6, 1e6)

        return z

    def unlift(self, z: np.ndarray) -> np.ndarray:
        """
        Extract original state from lifted representation.

        The first state_dim features are the linear terms [x_1, x_2, ..., x_n].
        """
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return z[:, :self.state_dim].copy()

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Single-step prediction.

        Args:
            x: Current state [batch, state_dim]
            u: Control [batch, u_dim]

        Returns:
            x_next: Next state [batch, state_dim]
        """
        with np.errstate(all='ignore'):
            z = self.lift(x)
            z_next = z @ self.A.T + u.reshape(-1, self.u_dim) @ self.B.T

            if not np.all(np.isfinite(z_next)):
                z_next = np.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=-1e6)

        return self.unlift(z_next)

    def predict_ksteps(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """
        Multi-step autoregressive prediction.

        Args:
            x0: Initial state [batch, state_dim]
            u_seq: Control sequence [K, batch, u_dim]

        Returns:
            x_pred: Predicted states [K, batch, state_dim]
        """
        K, batch, _ = u_seq.shape
        x_pred = np.zeros((K, batch, self.state_dim), dtype=np.float64)

        x = x0.copy()
        for k in range(K):
            x = self.predict(x, u_seq[k])
            x = np.clip(x, -1e6, 1e6)
            x_pred[k] = x

        return x_pred

    def save(self, path: str):
        """Save EDMD model to file."""
        np.savez(path,
                 A=self.A, B=self.B,
                 state_dim=self.state_dim, u_dim=self.u_dim,
                 encode_dim_param=self.encode_dim_param,
                 max_degree=self.max_degree, lift_dim=self.lift_dim,
                 target_dim=self.target_dim,
                 feature_specs=np.array(self.feature_specs, dtype=np.int32))

    @classmethod
    def load(cls, path: str) -> 'EDMD':
        """Load EDMD model from file."""
        data = np.load(path, allow_pickle=True)
        model = cls(int(data['state_dim']), int(data['u_dim']),
                    int(data['encode_dim_param']))
        model.A = data['A']
        model.B = data['B']
        model.max_degree = int(data['max_degree'])
        model.lift_dim = int(data['lift_dim'])
        if 'target_dim' in data:
            model.target_dim = int(data['target_dim'])
        if 'feature_specs' in data:
            model.feature_specs = [tuple(spec) for spec in data['feature_specs']]
        model.fitted = True
        return model

    def count_parameters(self) -> int:
        """Count number of parameters (A and B matrices)."""
        if self.A is None:
            return 0
        return self.A.size + self.B.size


# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_mlp_from_checkpoint(checkpoint_path: str,
                             device: torch.device = None) -> Tuple[MLPDynamics, Optional[Dict]]:
    """
    Load MLP model from saved checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: torch device (default: cpu)

    Returns:
        model: Loaded MLPDynamics model
        norm_stats: Normalization statistics dict (or None)
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim = checkpoint['state_dim']
    u_dim = checkpoint['u_dim']
    hidden_dim = checkpoint.get('hidden_dim', 256)
    hidden_layers = checkpoint.get('hidden_layers', 3)

    model = MLPDynamics(state_dim, u_dim, hidden_dim, hidden_layers)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    # Extract normalization stats if available
    norm_stats = checkpoint.get('norm_stats', None)

    return model, norm_stats


def load_edmd_from_checkpoint(checkpoint_path: str) -> Tuple[EDMD, Optional[Dict]]:
    """
    Load EDMD model from saved checkpoint (.pt or .npz).

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        model: Loaded EDMD model
        norm_stats: Normalization statistics dict (or None)
    """
    norm_stats = None

    if checkpoint_path.endswith('.npz'):
        return EDMD.load(checkpoint_path), None

    # Load from .pt file
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dim = checkpoint['state_dim']
    u_dim = checkpoint['u_dim']
    encode_dim_param = checkpoint['encode_dim_param']

    model = EDMD(state_dim, u_dim, encode_dim_param)

    # Load matrices
    model.A = checkpoint['A'].numpy() if isinstance(checkpoint['A'], torch.Tensor) else checkpoint['A']
    model.B = checkpoint['B'].numpy() if isinstance(checkpoint['B'], torch.Tensor) else checkpoint['B']
    model.max_degree = checkpoint.get('max_degree', 2)
    model.lift_dim = checkpoint.get('lift_dim', model.target_dim)

    if 'target_dim' in checkpoint:
        model.target_dim = checkpoint['target_dim']
    if 'feature_specs' in checkpoint:
        model.feature_specs = checkpoint['feature_specs']

    model.fitted = True

    # Extract normalization stats if available
    norm_stats = checkpoint.get('norm_stats', None)

    return model, norm_stats


def get_baseline_model_path(env_name: str, seed: int, encode_dim_param: int,
                            train_samples: int, method: str,
                            log_dir: str = None) -> str:
    """
    Get path to saved baseline model.

    Args:
        env_name: Environment name
        seed: Random seed
        encode_dim_param: Encode dimension parameter
        train_samples: Number of training samples
        method: 'mlp' or 'edmd'
        log_dir: Base log directory

    Returns:
        Full path to model checkpoint
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__),
                               '../../log/BaselineResults_Full_seed1/best_models')

    config_key = f"{env_name}_{seed}_{encode_dim_param}_{train_samples}"
    return os.path.join(log_dir, f"{method.lower()}_{config_key}.pt")


def load_baseline_results_csv(log_dir: str = None) -> pd.DataFrame:
    """
    Load baseline_results.csv with model information.

    Args:
        log_dir: Base log directory

    Returns:
        DataFrame with model info
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(__file__),
                               '../../log/BaselineResults_Full_seed1')

    csv_path = os.path.join(log_dir, 'baseline_results.csv')
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    return pd.read_csv(csv_path)


def select_best_baseline_models(df: pd.DataFrame, env_name: str, method: str,
                                n_best: int = 3,
                                largest_only: bool = False) -> pd.DataFrame:
    """
    Select best N models by test_Kloss for each configuration.

    Args:
        df: DataFrame with model info
        env_name: Environment name to filter
        method: 'MLP' or 'EDMD'
        n_best: Number of best models to select per configuration
        largest_only: If True, only select from largest train_samples and encode_dim_param

    Returns:
        Filtered DataFrame with selected models
    """
    # Filter by env and method
    df = df[(df['env_name'] == env_name) & (df['method'] == method)].copy()

    if len(df) == 0:
        return pd.DataFrame()

    if largest_only:
        max_samples = df['train_samples'].max()
        max_dim = df['encode_dim_param'].max()
        df = df[(df['train_samples'] == max_samples) &
                (df['encode_dim_param'] == max_dim)]
        return df.sort_values('test_Kloss').head(n_best)

    # Select best per (train_samples, encode_dim_param)
    selected = []
    for train_samples in df['train_samples'].unique():
        for encode_dim_param in df['encode_dim_param'].unique():
            subset = df[(df['train_samples'] == train_samples) &
                       (df['encode_dim_param'] == encode_dim_param)]
            if len(subset) > 0:
                selected.append(subset.sort_values('test_Kloss').head(n_best))

    if not selected:
        return pd.DataFrame()

    return pd.concat(selected, ignore_index=True)


# ============================================================================
# Normalization Utilities
# ============================================================================

def normalize_state(state: np.ndarray, norm_stats: Optional[Dict]) -> np.ndarray:
    """
    Normalize state using training statistics.

    Args:
        state: State array
        norm_stats: Dict with 'state_mean' and 'state_std' keys

    Returns:
        Normalized state
    """
    if norm_stats is None:
        return state
    state_mean = np.array(norm_stats['state_mean'])
    state_std = np.array(norm_stats['state_std'])
    state_std = np.maximum(state_std, 1e-8)  # Prevent division by zero
    return (state - state_mean) / state_std


def denormalize_state(state: np.ndarray, norm_stats: Optional[Dict]) -> np.ndarray:
    """
    Denormalize state to physical units.

    Args:
        state: Normalized state array
        norm_stats: Dict with 'state_mean' and 'state_std' keys

    Returns:
        Physical state
    """
    if norm_stats is None:
        return state
    state_mean = np.array(norm_stats['state_mean'])
    state_std = np.array(norm_stats['state_std'])
    return state * state_std + state_mean


def denormalize_action(action: np.ndarray, norm_stats: Optional[Dict]) -> np.ndarray:
    """
    Denormalize action to physical units.

    Args:
        action: Normalized action array
        norm_stats: Dict with 'action_mean' and 'action_std' keys

    Returns:
        Physical action
    """
    if norm_stats is None:
        return action
    action_mean = np.array(norm_stats['action_mean'])
    action_std = np.array(norm_stats['action_std'])
    return action * action_std + action_mean


def normalize_action(action: np.ndarray, norm_stats: Optional[Dict]) -> np.ndarray:
    """
    Normalize action using training statistics.

    Args:
        action: Physical action array
        norm_stats: Dict with 'action_mean' and 'action_std' keys

    Returns:
        Normalized action
    """
    if norm_stats is None:
        return action
    action_mean = np.array(norm_stats['action_mean'])
    action_std = np.array(norm_stats['action_std'])
    action_std = np.maximum(action_std, 1e-8)
    return (action - action_mean) / action_std