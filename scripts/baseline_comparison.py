#!/usr/bin/env python3
"""
Baseline Comparison for Koopman Neural Network Paper
=====================================================

Compares two baseline methods against Koopman neural networks:
1. MLP Dynamics: Standard feedforward network predicting x_{t+1} from (x_t, u_t)
2. EDMD: Extended Dynamic Mode Decomposition with hierarchical polynomial lifting

Experimental Grid:
- Environments: G1, Go2, FrankaImproved, Kinova
- Encode dim params: [1, 4, 16] (multiplier for state_dim, matching Koopman)
- Training samples: [1000, 16000, 140000]
- Seeds: [17382, 76849, 20965]

Dimension Matching Strategy:
- Koopman: Nkoopman = state_dim * (1 + encode_dim_param)
- MLP: Uses same encoder architecture (3 layers, 256 hidden) as Koopman
- EDMD: Hierarchical polynomial features built to exactly match Koopman dimension:
  1. Linear terms: [x_1, x_2, ..., x_n]
  2. 2nd order powers: [x_1^2, x_2^2, ...]
  3. 3rd order powers: [x_1^3, x_2^3, ...]
  4. 2nd order cross terms: [x_i*x_j for i < j]
  5. 3rd order cross terms: [x_i^2*x_j, x_i*x_j*x_k]
  6. 4th+ order if needed

For each (env, seed): 5 unique configurations
- 3 encode_dim_params with max train_samples
- 3 train_samples with max encode_dim_param (overlap = 1)

Output:
- baseline_results.csv: All experimental results
- baseline_models/: Saved model checkpoints
- baseline_comparison.pdf: Paper-quality comparison figure
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl

# Add utility path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utility'))

# Environment dimensions (pre-defined to avoid pybullet import issues)
ENV_DIMS = {
    "G1": {"state_dim": 53, "u_dim": 23},
    "Go2": {"state_dim": 35, "u_dim": 12},
    "Franka": {"state_dim": 14, "u_dim": 7},
    "FrankaImproved": {"state_dim": 14, "u_dim": 7},
    "Kinova": {"state_dim": 14, "u_dim": 7},
    "KinovaImproved": {"state_dim": 14, "u_dim": 7},
    "DampingPendulum": {"state_dim": 2, "u_dim": 1},
    "DoublePendulum": {"state_dim": 4, "u_dim": 2},
    "DoublePendulumImproved": {"state_dim": 4, "u_dim": 2},
}

# Environment-specific configurations (matching train_model.py)
# Environments that use normalization and gamma=0.99
# Note: Based on Koopman results CSV, Franka also uses gamma=0.99 and normalize=True
NORMALIZED_ENVS = {'G1', 'Go2', 'G1Improved', 'Go2Improved',  'FrankaImproved', 'KinovaImproved'}
# All other environments use normalize=False and gamma=0.8

def get_gamma(env_name: str) -> float:
    """Get gamma value for environment (matching train_model.py)."""
    if env_name in NORMALIZED_ENVS:
        return 0.99
    else:
        return 0.8

# Try importing the data collector, but provide fallback for direct loading
try:
    from dataset import KoopmanDatasetCollector
    HAS_DATASET_COLLECTOR = True
except (ImportError, AttributeError) as e:
    print(f"Warning: Could not import KoopmanDatasetCollector: {e}")
    print("Will use direct .pt file loading instead.")
    HAS_DATASET_COLLECTOR = False
    KoopmanDatasetCollector = None

try:
    from network import KoopmanNet
except (ImportError, AttributeError):
    KoopmanNet = None


def load_cached_dataset(env_name, train_samples, val_samples, test_samples, Ksteps, seed):
    """
    Load pre-cached dataset from .pt file.

    Applies normalization based on environment (matching train_model.py):
    - G1, Go2, FrankaImproved, KinovaImproved: normalize=True
    - Others: normalize=False

    Returns:
        train_data, val_data, test_data, state_dim, u_dim, norm_stats
        or None if dataset not found
    """
    project_root = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(project_root, "data", "datasets")

    # Try different max_train_samples values and filename patterns
    for max_train in [140000, 64000, 16000]:
        # Try different filename patterns (some datasets use 'steps' instead of 'Ksteps')
        filename_patterns = [
            f"{env_name}_train{max_train}_val{val_samples}_test{test_samples}_Ksteps{Ksteps}.pt",
            f"{env_name}_train{max_train}_val{val_samples}_test{test_samples}_steps{Ksteps}.pt",
        ]

        for filename in filename_patterns:
            dataset_path = os.path.join(data_dir, filename)
            if os.path.exists(dataset_path):
                print(f"  Loading cached dataset: {dataset_path}")
                try:
                    checkpoint = torch.load(dataset_path, map_location='cpu', weights_only=False)
                except Exception as e:
                    print(f"  Error loading .pt file: {e}")
                    continue  # Try next pattern

                # Handle both key formats: 'train_data' and 'Ktrain_data'
                if 'train_data' in checkpoint:
                    train_data = checkpoint['train_data']
                    val_data = checkpoint['val_data']
                    test_data = checkpoint['test_data']
                elif 'Ktrain_data' in checkpoint:
                    train_data = checkpoint['Ktrain_data']
                    val_data = checkpoint['Kval_data']
                    test_data = checkpoint['Ktest_data']
                else:
                    print(f"  Unknown checkpoint format: {list(checkpoint.keys())[:5]}")
                    continue

                # Get dimensions
                if env_name in ENV_DIMS:
                    state_dim = ENV_DIMS[env_name]["state_dim"]
                    u_dim = ENV_DIMS[env_name]["u_dim"]
                else:
                    # Infer from data shape: [steps, traj, u_dim + state_dim]
                    total_dim = train_data.shape[2]
                    state_dim = checkpoint.get('state_dim', total_dim // 2)
                    u_dim = total_dim - state_dim

                # Convert to numpy
                if isinstance(train_data, torch.Tensor):
                    train_data = train_data.numpy()
                if isinstance(val_data, torch.Tensor):
                    val_data = val_data.numpy()
                if isinstance(test_data, torch.Tensor):
                    test_data = test_data.numpy()

                # Subsample training data if needed
                n_train = train_data.shape[1]
                if train_samples < n_train:
                    np.random.seed(seed)
                    indices = np.random.choice(n_train, train_samples, replace=False)
                    train_data = train_data[:, indices, :]

                # Apply normalization based on environment (matching train_model.py)
                norm_stats = None
                if env_name in NORMALIZED_ENVS:
                    print(f"  Applying normalization for {env_name} (matching train_model.py)")
                    # Compute normalization stats from training data
                    action_train_mean = np.mean(train_data[..., :u_dim], axis=(0, 1))
                    action_train_std = np.std(train_data[..., :u_dim], axis=(0, 1))
                    state_train_mean = np.mean(train_data[..., u_dim:], axis=(0, 1))
                    state_train_std = np.std(train_data[..., u_dim:], axis=(0, 1))

                    # Prevent division by zero
                    action_train_std = np.maximum(action_train_std, 1e-8)
                    state_train_std = np.maximum(state_train_std, 1e-8)

                    # Apply normalization
                    train_data[..., :u_dim] = (train_data[..., :u_dim] - action_train_mean) / action_train_std
                    train_data[..., u_dim:] = (train_data[..., u_dim:] - state_train_mean) / state_train_std
                    val_data[..., :u_dim] = (val_data[..., :u_dim] - action_train_mean) / action_train_std
                    val_data[..., u_dim:] = (val_data[..., u_dim:] - state_train_mean) / state_train_std
                    test_data[..., :u_dim] = (test_data[..., :u_dim] - action_train_mean) / action_train_std
                    test_data[..., u_dim:] = (test_data[..., u_dim:] - state_train_mean) / state_train_std

                    norm_stats = {
                        'action_mean': action_train_mean,
                        'action_std': action_train_std,
                        'state_mean': state_train_mean,
                        'state_std': state_train_std,
                    }
                else:
                    print(f"  No normalization for {env_name} (matching train_model.py)")

                return train_data, val_data, test_data, state_dim, u_dim, norm_stats

    return None


def generate_synthetic_data(env_name, train_samples, val_samples, test_samples, Ksteps, seed):
    """
    Generate synthetic data for testing when real data is unavailable.

    This creates simple stable linear dynamics with noise to test the baseline methods.
    Returns data in the same format as load_cached_dataset.
    """
    np.random.seed(seed)

    if env_name not in ENV_DIMS:
        return None

    state_dim = ENV_DIMS[env_name]["state_dim"]
    u_dim = ENV_DIMS[env_name]["u_dim"]

    print(f"  Generating synthetic data for {env_name} (state_dim={state_dim}, u_dim={u_dim})")

    # Create stable linear dynamics: x_{t+1} = A @ x_t + B @ u_t + noise
    # Use diagonal-dominant matrix for guaranteed stability
    A = np.eye(state_dim) * 0.9 + np.random.randn(state_dim, state_dim) * 0.05
    # Ensure stability by scaling
    eigenvalues = np.linalg.eigvals(A)
    max_eig = np.max(np.abs(eigenvalues))
    if max_eig > 0.95:
        A = A * 0.9 / max_eig

    B = np.random.randn(state_dim, u_dim) * 0.05

    def generate_trajectories(n_traj, steps):
        """Generate trajectory data."""
        data = np.zeros((steps, n_traj, u_dim + state_dim), dtype=np.float32)

        for i in range(n_traj):
            x = np.random.randn(state_dim) * 0.2  # Smaller initial states

            for t in range(steps):
                u = np.random.randn(u_dim) * 0.2  # Smaller control inputs

                # Store [u, x]
                data[t, i, :u_dim] = u
                data[t, i, u_dim:] = x

                # Update state with clipping for stability
                x_new = A @ x + B @ u + np.random.randn(state_dim) * 0.005
                x = np.clip(x_new, -5, 5)  # Prevent explosion

        return data

    steps = Ksteps + 1
    train_data = generate_trajectories(train_samples, steps)
    val_data = generate_trajectories(val_samples, steps)
    test_data = generate_trajectories(test_samples, steps)

    return train_data, val_data, test_data, state_dim, u_dim


def get_available_datasets(envs, val_samples=20000, test_samples=20000, Ksteps=15):
    """Check which environments have cached datasets."""
    available = []
    project_root = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(project_root, "data", "datasets")

    for env_name in envs:
        found = False
        for max_train in [140000, 64000, 16000]:
            # Try both filename patterns
            patterns = [
                f"{env_name}_train{max_train}_val{val_samples}_test{test_samples}_Ksteps{Ksteps}.pt",
                f"{env_name}_train{max_train}_val{val_samples}_test{test_samples}_steps{Ksteps}.pt",
            ]
            for pattern in patterns:
                dataset_path = os.path.join(data_dir, pattern)
                if os.path.exists(dataset_path):
                    available.append(env_name)
                    found = True
                    break
            if found:
                break

    return available

# ============================================================================
# Paper-Quality Plot Settings
# ============================================================================
mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "figure.constrained_layout.use": True,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.4,
    "lines.markersize": 5.5,
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
})

# ============================================================================
# Configuration
# ============================================================================
# Full experiment configuration for all environments
# Note: Use 'Franka' (not FrankaImproved) to match Koopman results CSV
ENVS = ["G1", "Go2", "FrankaImproved", "Kinova", "DoublePendulum", "DampingPendulum"]
SEEDS = [17382, 76849, 20965]

# Experimental grid - matching Koopman experiments
# Using encode_dim_param (multiplier) instead of absolute hidden dims
ENCODE_DIM_PARAMS = [1, 4, 16]            # encode_dim = encode_dim_param * state_dim
TRAIN_SAMPLES_LIST = [1000, 16000, 140000]  # Training sample sizes
HIDDEN_LAYERS = 3                          # Fixed number of hidden layers for MLP encoder (matching Koopman)
HIDDEN_DIM = 256                           # Fixed hidden dimension for MLP encoder (matching Koopman)

# Fixed parameters
VAL_SAMPLES = 20000
TEST_SAMPLES = 20000
KSTEPS = 15                              # Multi-step prediction horizon
BATCH_SIZE = 64                          # Match Koopman training batch size (use 256+ for faster GPU training)
MAX_TRAIN_SAMPLES = 140000               # Maximum for data collection
NUM_WORKERS = 4                          # DataLoader workers (0 for debugging)

# Training parameters - matching train_model.py iteration-level training
TRAIN_STEPS = 60000                      # Total training iterations (same as Koopman default)
TRAIN_STEPS_FAST = 10000                 # Fast mode: reduced iterations for testing
VAL_STEP = 1000                          # Validate every N steps
LEARNING_RATE = 1e-3
LR_STEP = 1000                           # LR decay step (iterations)
LR_GAMMA = 0.9                           # LR decay factor (matching Koopman training)
MAX_NORM = 0.1                           # Gradient clipping max norm (matching train_model.py)
EARLY_STOP_PATIENCE = 20                 # Number of val checks without improvement (matching Koopman)
EARLY_STOP_MIN_DELTA = 1e-8              # Minimum improvement to reset patience (matching Koopman)
MINIMUM_TRAIN_STEPS = 20000              # Only consider early stopping after this (matching Koopman)
MINIMUM_TRAIN_STEPS_FAST = 5000          # Fast mode: reduced minimum


def get_koopman_dims(state_dim, encode_dim_param):
    """
    Compute dimensions matching Koopman network.

    In Koopman experiments:
    - actual_encode_dim = encode_dim_param * state_dim
    - Nkoopman = state_dim + actual_encode_dim = state_dim * (1 + encode_dim_param)

    Returns:
        actual_encode_dim: The latent dimension added by encoder
        total_koopman_dim: Total Koopman state dimension (state_dim + actual_encode_dim)
    """
    actual_encode_dim = encode_dim_param * state_dim
    total_koopman_dim = state_dim + actual_encode_dim
    return actual_encode_dim, total_koopman_dim


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
    Uses SVD-based numerically stable solver for A and B matrices.
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
        # High-dim systems (>40 states) with high encode_dim_param can explode
        max_safe_dim = min(self.target_dim, 5000)  # Cap at 5000 features max
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

        Returns:
            List of exponent tuples defining each polynomial feature.
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
        # First: x_i^2 * x_j
        for i in range(n):
            for j in range(n):
                if i != j:
                    exp = [0] * n
                    exp[i] = 2
                    exp[j] = 1
                    specs.append(tuple(exp))
                    if len(specs) >= target:
                        return specs

        # Then: x_i * x_j * x_k for i < j < k
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

        # 7. Fourth order cross terms (x_i^3 * x_j, x_i^2 * x_j^2, etc.)
        # x_i^3 * x_j
        for i in range(n):
            for j in range(n):
                if i != j:
                    exp = [0] * n
                    exp[i] = 3
                    exp[j] = 1
                    specs.append(tuple(exp))
                    if len(specs) >= target:
                        return specs

        # x_i^2 * x_j^2
        for i in range(n):
            for j in range(i + 1, n):
                exp = [0] * n
                exp[i] = 2
                exp[j] = 2
                specs.append(tuple(exp))
                if len(specs) >= target:
                    return specs

        # x_i^2 * x_j * x_k
        for i in range(n):
            for j in range(n):
                for k in range(j + 1, n):
                    if i != j and i != k:
                        exp = [0] * n
                        exp[i] = 2
                        exp[j] = 1
                        exp[k] = 1
                        specs.append(tuple(exp))
                        if len(specs) >= target:
                            return specs

        # x_i * x_j * x_k * x_l for i < j < k < l
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        exp = [0] * n
                        exp[i] = 1
                        exp[j] = 1
                        exp[k] = 1
                        exp[l] = 1
                        specs.append(tuple(exp))
                        if len(specs) >= target:
                            return specs

        # If still not enough, add 5th order powers
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

        Uses precomputed feature specifications for efficient computation.
        Applies numerical stability measures (clipping, safe power computation).

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
            # Compute product: x_1^e_1 * x_2^e_2 * ... * x_n^e_n
            feat = np.ones(batch, dtype=np.float64)
            for var_idx, exp in enumerate(exponents):
                if exp > 0:
                    # Use safe power computation
                    feat *= np.power(x_safe[:, var_idx], exp)

            # Clip result for numerical stability
            z[:, feat_idx] = np.clip(feat, -1e6, 1e6)

        return z

    def unlift(self, z: np.ndarray) -> np.ndarray:
        """
        Extract original state from lifted representation.

        The first state_dim features are the linear terms [x_1, x_2, ..., x_n].

        Args:
            z: Lifted state [batch, lift_dim]

        Returns:
            x: Original state [batch, state_dim]
        """
        if z.ndim == 1:
            z = z.reshape(1, -1)

        # First state_dim features are linear terms
        return z[:, :self.state_dim].copy()

    def fit(self, X: np.ndarray, U: np.ndarray, X_next: np.ndarray,
            regularization: float = 1e-4):
        """
        Fit EDMD model using numerically stable least squares solver.

        Uses regularized least squares with fallback options for numerical stability.

        Args:
            X: States [N, state_dim]
            U: Controls [N, u_dim]
            X_next: Next states [N, state_dim]
            regularization: Tikhonov regularization parameter (larger = more stable)
        """
        with np.errstate(all='ignore'):  # Suppress warnings during fitting
            # Lift current and next states
            Z = self.lift(X)
            Z_next = self.lift(X_next)

            # Check for NaN/Inf in lifted features
            has_nan_Z = not np.all(np.isfinite(Z))
            has_nan_Z_next = not np.all(np.isfinite(Z_next))

            if has_nan_Z or has_nan_Z_next:
                print(f"    Warning: NaN/Inf in lifted features (Z: {has_nan_Z}, Z_next: {has_nan_Z_next})")
                Z = np.nan_to_num(Z, nan=0.0, posinf=1e6, neginf=-1e6)
                Z_next = np.nan_to_num(Z_next, nan=0.0, posinf=1e6, neginf=-1e6)

            # Concatenate lifted state and control: [Z, U]
            ZU = np.hstack([Z, U])

            # Use regularized least squares (more stable than SVD for ill-conditioned problems)
            n_features = ZU.shape[1]

            # Method 1: Regularized normal equations (most stable for our use case)
            try:
                reg_matrix = regularization * np.eye(n_features)
                Theta = np.linalg.solve(
                    ZU.T @ ZU + reg_matrix,
                    ZU.T @ Z_next
                ).T
            except np.linalg.LinAlgError as e:
                print(f"    Normal equations failed ({e}), trying lstsq")
                # Method 2: Use lstsq with rcond
                try:
                    Theta, _, _, _ = np.linalg.lstsq(ZU, Z_next, rcond=max(regularization, 1e-6))
                    Theta = Theta.T
                except np.linalg.LinAlgError as e2:
                    print(f"    lstsq also failed ({e2}), using pinv")
                    # Method 3: Use pseudoinverse (last resort)
                    try:
                        ZU_pinv = np.linalg.pinv(ZU, rcond=max(regularization, 1e-6))
                        Theta = (ZU_pinv @ Z_next).T
                    except np.linalg.LinAlgError:
                        print("    All solvers failed, using zero matrices")
                        Theta = np.zeros((self.lift_dim, self.lift_dim + self.u_dim))

            # Final sanity check
            if not np.all(np.isfinite(Theta)):
                print("    Warning: Non-finite values in Theta, replacing with zeros")
                Theta = np.nan_to_num(Theta, nan=0.0, posinf=0.0, neginf=0.0)

        self.A = Theta[:, :self.lift_dim]
        self.B = Theta[:, self.lift_dim:]
        self.fitted = True

    def predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Single-step prediction.

        Args:
            x: Current state [batch, state_dim]
            u: Control [batch, u_dim]

        Returns:
            x_next: Next state [batch, state_dim]
        """
        with np.errstate(all='ignore'):  # Suppress warnings for numerical issues
            z = self.lift(x)
            z_next = z @ self.A.T + u @ self.B.T

            # Handle numerical issues
            if not np.all(np.isfinite(z_next)):
                z_next = np.nan_to_num(z_next, nan=0.0, posinf=1e6, neginf=-1e6)

        # Extract state from lifted representation
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
            # Clip to prevent explosion
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

def load_mlp_from_checkpoint(checkpoint_path: str, device: torch.device = None) -> MLPDynamics:
    """
    Load MLP model from saved checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: torch device (default: cpu)

    Returns:
        Loaded MLPDynamics model
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dim = checkpoint['state_dim']
    u_dim = checkpoint['u_dim']
    hidden_dim = checkpoint.get('hidden_dim', 256)
    hidden_layers = checkpoint.get('hidden_layers', 2)

    model = MLPDynamics(state_dim, u_dim, hidden_dim, hidden_layers)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    return model


def load_edmd_from_checkpoint(checkpoint_path: str) -> EDMD:
    """
    Load EDMD model from saved checkpoint (.pt or .npz).

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded EDMD model
    """
    if checkpoint_path.endswith('.npz'):
        return EDMD.load(checkpoint_path)

    # Load from .pt file
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    state_dim = checkpoint['state_dim']
    u_dim = checkpoint['u_dim']
    encode_dim_param = checkpoint['encode_dim_param']

    model = EDMD(state_dim, u_dim, encode_dim_param)

    # Load matrices
    model.A = checkpoint['A'].numpy() if isinstance(checkpoint['A'], torch.Tensor) else checkpoint['A']
    model.B = checkpoint['B'].numpy() if isinstance(checkpoint['B'], torch.Tensor) else checkpoint['B']
    model.max_degree = checkpoint['max_degree']
    model.lift_dim = checkpoint['lift_dim']

    if 'target_dim' in checkpoint:
        model.target_dim = checkpoint['target_dim']
    if 'feature_specs' in checkpoint:
        model.feature_specs = checkpoint['feature_specs']

    model.fitted = True

    return model


def get_baseline_model_path(env_name: str, seed: int, encode_dim_param: int,
                            train_samples: int, method: str, log_dir: str = None) -> str:
    """
    Get path to saved baseline model.

    Args:
        env_name: Environment name
        seed: Random seed
        encode_dim_param: Encode dimension parameter
        train_samples: Number of training samples
        method: 'mlp' or 'edmd'
        log_dir: Base log directory (default: log/Baseline_Fullresult)

    Returns:
        Path to model checkpoint
    """
    if log_dir is None:
        project_root = os.path.join(os.path.dirname(__file__), '..')
        log_dir = os.path.join(project_root, 'log', 'Fullresult_baseline')

    model_dir = os.path.join(log_dir, 'best_models')
    config_key = f"{env_name}_{seed}_{encode_dim_param}_{train_samples}"

    return os.path.join(model_dir, f"{method.lower()}_{config_key}.pt")


# ============================================================================
# Training Functions
# ============================================================================

def train_mlp(model: MLPDynamics, train_data: np.ndarray, val_data: np.ndarray,
              u_dim: int, device: torch.device, gamma: float = 0.8,
              train_steps: int = TRAIN_STEPS, batch_size: int = BATCH_SIZE,
              num_workers: int = NUM_WORKERS,
              val_step: int = VAL_STEP, lr: float = LEARNING_RATE,
              lr_step: int = LR_STEP, lr_gamma: float = LR_GAMMA,
              early_stop_patience: int = EARLY_STOP_PATIENCE,
              early_stop_min_delta: float = EARLY_STOP_MIN_DELTA,
              minimum_train_steps: int = MINIMUM_TRAIN_STEPS,
              use_compile: bool = False) -> tuple:
    """
    Train MLP dynamics model with iteration-level training (matching Koopman training).

    Uses the same training loop structure as train_model.py:
    - Iteration-based training (not epoch-based)
    - StepLR scheduler with per-iteration updates
    - Validation every val_step iterations
    - Early stopping based on validation checks

    Args:
        model: MLPDynamics model to train
        train_data: Training data [steps, traj, u_dim + state_dim]
        val_data: Validation data
        u_dim: Control dimension
        device: torch device
        train_steps: Total training iterations
        val_step: Validate every N steps
        lr: Initial learning rate
        lr_step: LR decay step size (iterations)
        lr_gamma: LR decay factor
        early_stop_patience: Validation checks without improvement before stopping
        early_stop_min_delta: Minimum improvement to reset patience
        minimum_train_steps: Only consider early stopping after this many steps

    Returns:
        model: Trained model
        val_losses: List of validation losses
        best_val_loss: Best validation loss achieved
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)
    mse_loss = nn.MSELoss()

    # Prepare training data: predict x_{t+1} from (x_t, u_t)
    steps, traj, dim = train_data.shape
    state_dim = dim - u_dim

    # Flatten trajectory data for single-step prediction training
    X = train_data[:-1, :, u_dim:].reshape(-1, state_dim).astype(np.float32)
    U = train_data[:-1, :, :u_dim].reshape(-1, u_dim).astype(np.float32)
    X_next = train_data[1:, :, u_dim:].reshape(-1, state_dim).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(U),
        torch.from_numpy(X_next)
    )
    # Use persistent_workers for faster data loading when num_workers > 0
    use_persistent = num_workers > 0
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == 'cuda'),
                              persistent_workers=use_persistent, drop_last=True)

    # Optional: Use torch.compile for PyTorch 2.0+ (can provide 10-30% speedup)
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("    Using torch.compile() for faster training")
        except Exception as e:
            print(f"    torch.compile() failed: {e}, using eager mode")

    best_val_loss = float('inf')
    best_model_state = None
    val_losses = []
    step = 0

    # Early stopping variables
    patience_counter = 0
    early_stopped = False

    # Timing variables
    last_val_time = time.time()
    training_start_time = time.time()

    while step < train_steps and not early_stopped:
        for batch_x, batch_u, batch_x_next in train_loader:
            if step >= train_steps:
                break

            batch_x = batch_x.to(device)
            batch_u = batch_u.to(device)
            batch_x_next = batch_x_next.to(device)

            # Forward pass
            model.train()
            optimizer.zero_grad()
            x_pred = model(batch_x, batch_u)
            loss = mse_loss(x_pred, batch_x_next)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            optimizer.step()
            scheduler.step()

            # Validation
            if step % val_step == 0:
                model.eval()
                with torch.no_grad():
                    val_loss = evaluate_kstep_error(model, val_data, u_dim, device, gamma)

                # Early stopping check
                if val_loss < best_val_loss - early_stop_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    improved = "*"
                else:
                    patience_counter += 1
                    improved = ""
                    if step >= minimum_train_steps and patience_counter >= early_stop_patience:
                        print(f"    Early stopping at step {step}: no improvement for "
                              f"{early_stop_patience} validation checks")
                        early_stopped = True
                        break

                val_losses.append(val_loss)

                # Timing
                current_time = time.time()
                elapsed_since_last_val = current_time - last_val_time
                total_elapsed = current_time - training_start_time
                steps_per_sec = val_step / elapsed_since_last_val if elapsed_since_last_val > 0 else 0
                last_val_time = current_time
                current_lr = optimizer.param_groups[0]['lr']

                print(f"    Step {step:5d}/{train_steps}: val={val_loss:.6f}{improved}, "
                      f"lr={current_lr:.1e}, patience={patience_counter}/{early_stop_patience}, "
                      f"time={elapsed_since_last_val:.1f}s ({steps_per_sec:.1f} steps/s, "
                      f"total: {total_elapsed:.1f}s)")

            step += 1

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, val_losses, best_val_loss


def train_edmd(train_data: np.ndarray, u_dim: int, encode_dim_param: int = 1) -> EDMD:
    """
    Train EDMD model with hierarchical polynomial lifting.

    Args:
        train_data: Training data [steps, traj, u_dim + state_dim]
        u_dim: Control dimension
        encode_dim_param: Multiplier to match Koopman dimension

    Returns:
        model: Fitted EDMD model
    """
    steps, traj, dim = train_data.shape
    state_dim = dim - u_dim

    # Flatten data
    X = train_data[:-1, :, u_dim:].reshape(-1, state_dim).astype(np.float64)
    U = train_data[:-1, :, :u_dim].reshape(-1, u_dim).astype(np.float64)
    X_next = train_data[1:, :, u_dim:].reshape(-1, state_dim).astype(np.float64)

    model = EDMD(state_dim, u_dim, encode_dim_param=encode_dim_param)
    model.fit(X, U, X_next)

    return model


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_kstep_error(model, test_data: np.ndarray, u_dim: int,
                         device: torch.device, gamma: float = 0.8) -> float:
    """
    Evaluate K-step prediction error with gamma-weighted loss (matching Klinear_loss in train_model.py).

    Computes: loss = Σ(γ^i × MSE_i) / Σ(γ^i)

    This matches the exact evaluation used for Koopman models in train_model.py,
    making results directly comparable for the paper.

    Args:
        model: Trained model (MLPDynamics, EDMD, or KoopmanNet)
        test_data: Test data [steps, traj, u_dim + state_dim]
        u_dim: Control dimension
        device: torch device
        gamma: Discount factor for loss weighting (0.8 for most envs, 0.99 for G1/Go2/etc.)

    Returns:
        loss: Gamma-weighted MSE (same as test_Kloss in train_model.py)
    """
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.numpy()

    steps, traj, dim = test_data.shape
    state_dim = dim - u_dim
    K = steps - 1

    x0 = test_data[0, :, u_dim:]  # [traj, state_dim]
    u_seq = test_data[:-1, :, :u_dim]  # [K, traj, u_dim]
    x_gt = test_data[1:, :, u_dim:]  # [K, traj, state_dim]

    # Get predictions
    if isinstance(model, EDMD):
        x_pred = model.predict_ksteps(x0, u_seq)  # [K, traj, state_dim]
    elif KoopmanNet is not None and isinstance(model, KoopmanNet):
        model.eval()
        with torch.no_grad():
            x0_t = torch.from_numpy(x0).float().to(device)
            u_seq_t = torch.from_numpy(u_seq).float().to(device)

            z = model.encode(x0_t)
            x_pred = np.zeros((K, traj, state_dim), dtype=np.float64)
            for k in range(K):
                z = model.forward(z, u_seq_t[k])
                x_pred[k] = z[:, :state_dim].cpu().numpy()
    else:
        # MLP model
        model.eval()
        with torch.no_grad():
            x0_t = torch.from_numpy(x0.astype(np.float32)).to(device)
            u_seq_t = torch.from_numpy(u_seq.astype(np.float32)).to(device)

            x_pred_t = model.predict_ksteps(x0_t, u_seq_t)
            x_pred = x_pred_t.cpu().numpy()

    # Compute gamma-weighted loss (matching Klinear_loss in train_model.py)
    # loss = Σ(γ^i × MSE_i) / Σ(γ^i)
    beta = 1.0
    beta_sum = 0.0
    weighted_loss = 0.0

    for k in range(K):
        # MSE at step k (mean over trajectories and state dimensions)
        step_mse = np.mean((x_pred[k] - x_gt[k]) ** 2)
        weighted_loss += beta * step_mse
        beta_sum += beta
        beta *= gamma

    loss = weighted_loss / beta_sum
    return float(loss)


def load_koopman_results(env_name: str, train_samples: int, log_dir: str) -> float:
    """
    Load Koopman results from CSV for comparison.

    Returns best test_Kloss for the given environment and train_samples.
    """
    csv_path = os.path.join(log_dir, "all_result_with_improvedfranka.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(log_dir, "koopman_results_log.csv")

    if not os.path.exists(csv_path):
        return np.nan

    df = pd.read_csv(csv_path)

    # Map environment names
    lookup_env = env_name
    if env_name == "FrankaImproved":
        lookup_env = "Franka"  # FrankaImproved results stored as Franka

    # Filter for this environment and train_samples
    mask = (df['env_name'] == lookup_env) & (df['train_samples'] == train_samples)

    # Prefer best configuration (both losses enabled)
    best_mask = mask & (df['use_covariance_loss'] == True) & (df['use_control_loss'] == True)
    if best_mask.any():
        return df.loc[best_mask, 'test_Kloss'].min()
    elif mask.any():
        return df.loc[mask, 'test_Kloss'].min()
    else:
        return np.nan


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(output_dir: str, device_str: str = 'auto',
                   train_steps: int = TRAIN_STEPS, skip_existing: bool = False,
                   use_synthetic: bool = False, batch_size: int = BATCH_SIZE,
                   num_workers: int = NUM_WORKERS, use_compile: bool = False,
                   envs: list = None):
    """
    Run full baseline comparison experiment.

    Experimental design:
    - For each (env, seed): run 5 configurations
      - 3 encode_dim_params with max train_samples
      - 3 train_samples with max encode_dim_param (overlap = 1)

    This matches the Koopman experiment design where:
    - encode_dim = encode_dim_param * state_dim
    - Total Koopman dim = state_dim * (1 + encode_dim_param)
    """
    # Use provided envs or default to module-level ENVS
    env_list = envs if envs is not None else ENVS

    # Setup device
    if device_str == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Using device: {device}")

    # Create timestamped output directory: log/baseline_results/{timestamp}/
    project_root = os.path.join(os.path.dirname(__file__), '..')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Use output_dir as override, otherwise use timestamped directory
    if output_dir and not output_dir.startswith('BaselineResults'):
        # User provided explicit output path
        log_base_dir = output_dir
    else:
        # Default to timestamped directory
        log_base_dir = os.path.join(project_root, 'log', 'baseline_results', timestamp)

    model_dir = os.path.join(log_base_dir, "best_models")
    os.makedirs(log_base_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Output directory: {log_base_dir}")
    print(f"Model directory: {model_dir}")

    # Results storage - ALWAYS start fresh by default
    results = []
    results_csv_path = os.path.join(log_base_dir, "baseline_results.csv")

    # Load existing results only if skip_existing is True
    if skip_existing and os.path.exists(results_csv_path):
        existing_df = pd.read_csv(results_csv_path)
        results = existing_df.to_dict('records')
        print(f"Loaded {len(results)} existing results")
    else:
        # Clear existing results file
        if os.path.exists(results_csv_path):
            os.remove(results_csv_path)
            print("Cleared existing results file")

    # Koopman log directory for comparison
    koopman_log_dir = os.path.join(project_root, 'log', 'Fullresult')

    # Build experiment configurations
    # For each env/seed:
    #   - Vary encode_dim_param with train_samples = max (3 configs)
    #   - Vary train_samples with encode_dim_param = max (3 configs)
    #   - Overlap: (max_encode_dim_param, max_samples) counted once
    # Total per (env, seed): 3 + 3 - 1 = 5 configs

    max_encode_param = max(ENCODE_DIM_PARAMS)
    max_samples = max(TRAIN_SAMPLES_LIST)

    configs = []
    for encode_dim_param in ENCODE_DIM_PARAMS:
        configs.append((encode_dim_param, max_samples))
    for train_samples in TRAIN_SAMPLES_LIST:
        if train_samples != max_samples:  # Avoid duplicate
            configs.append((max_encode_param, train_samples))

    print(f"\nExperiment configuration:")
    print(f"  Environments: {env_list}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Encode dim params: {ENCODE_DIM_PARAMS} (multiplier for state_dim)")
    print(f"  Train samples: {TRAIN_SAMPLES_LIST}")
    print(f"  Configs per (env, seed): {len(configs)}")
    print(f"  Total experiments: {len(env_list) * len(SEEDS) * len(configs) * 2}")  # *2 for MLP and EDMD

    # Check which environments have cached datasets
    available_envs = get_available_datasets(env_list, VAL_SAMPLES, TEST_SAMPLES, KSTEPS)
    print(f"\n  Available cached datasets: {available_envs}")
    if not HAS_DATASET_COLLECTOR:
        missing_envs = [e for e in env_list if e not in available_envs]
        if missing_envs:
            print(f"  WARNING: No dataset collector available. Missing envs: {missing_envs}")
            print(f"  These environments will be skipped.")

    # Data cache to avoid reloading
    data_cache = {}

    total_start = time.time()

    for env_name in env_list:
        print(f"\n{'='*70}")
        print(f"Environment: {env_name}")
        print('='*70)

        state_dim = None
        u_dim = None
        gamma = get_gamma(env_name)  # Get gamma for this environment (matching train_model.py)
        print(f"  Using gamma={gamma} (matching train_model.py)")

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            # Set seeds
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            for encode_dim_param, train_samples in configs:
                config_key = f"{env_name}_{seed}_{encode_dim_param}_{train_samples}"

                # Check if already computed (only if skip_existing is True)
                if skip_existing:
                    existing = [r for r in results
                               if r['Environment'] == env_name
                               and r['Seed'] == seed
                               and r['EncodeDimParam'] == encode_dim_param
                               and r['TrainSamples'] == train_samples]

                    if len(existing) >= 2:  # MLP and EDMD
                        print(f"  Skipping {config_key} (already computed)")
                        continue

                # Compute dimensions for this configuration
                actual_encode_dim, total_koopman_dim = None, None

                print(f"\n  Config: encode_dim_param={encode_dim_param}, train_samples={train_samples}")

                # Load or get cached data
                data_key = f"{env_name}_{seed}_{train_samples}"
                if data_key not in data_cache:
                    print("  Loading data...")
                    try:
                        # First try direct cached loading (works without pybullet)
                        cached_result = load_cached_dataset(
                            env_name, train_samples, VAL_SAMPLES, TEST_SAMPLES, KSTEPS, seed
                        )
                        if cached_result is not None:
                            train_data, val_data, test_data, state_dim, u_dim, norm_stats = cached_result
                            data_cache[data_key] = (train_data, val_data, test_data, state_dim, u_dim, norm_stats)
                        elif HAS_DATASET_COLLECTOR and KoopmanDatasetCollector is not None:
                            # Fall back to data collector if available
                            # Use environment-specific normalization (matching train_model.py)
                            should_normalize = env_name in NORMALIZED_ENVS
                            data_collector = KoopmanDatasetCollector(
                                env_name,
                                train_samples=train_samples,
                                val_samples=VAL_SAMPLES,
                                test_samples=TEST_SAMPLES,
                                Ksteps=KSTEPS,
                                normalize=should_normalize,
                                seed=seed,
                                max_train_samples=MAX_TRAIN_SAMPLES
                            )
                            train_data, val_data, test_data = data_collector.get_data()
                            state_dim = data_collector.state_dim
                            u_dim = data_collector.u_dim
                            norm_stats = getattr(data_collector, 'norm_stats', None)
                            data_cache[data_key] = (train_data, val_data, test_data, state_dim, u_dim, norm_stats)
                        else:
                            # Always try synthetic data as last fallback (for corrupted datasets)
                            print(f"  No cached dataset or collector available, using synthetic data...")
                            synth_result = generate_synthetic_data(
                                env_name, train_samples, VAL_SAMPLES // 10, TEST_SAMPLES // 10, KSTEPS, seed
                            )
                            if synth_result is not None:
                                train_data, val_data, test_data, state_dim, u_dim = synth_result
                                norm_stats = None
                                data_cache[data_key] = (train_data, val_data, test_data, state_dim, u_dim, norm_stats)
                            else:
                                print(f"  ERROR: Cannot generate synthetic data for {env_name}.")
                                print(f"  Skipping environment {env_name}...")
                                break
                    except Exception as e:
                        print(f"  Error loading data: {e}")
                        continue
                else:
                    train_data, val_data, test_data, state_dim, u_dim, norm_stats = data_cache[data_key]

                # Compute Koopman-equivalent dimensions
                actual_encode_dim, total_koopman_dim = get_koopman_dims(state_dim, encode_dim_param)
                # MLP hidden dimension to match Koopman encoder capacity
                mlp_hidden_dim = HIDDEN_DIM  # Fixed hidden dim like Koopman encoder

                print(f"  State dim: {state_dim}, Control dim: {u_dim}")
                print(f"  Koopman equiv dims: encode_dim={actual_encode_dim}, total={total_koopman_dim}")
                print(f"  Train shape: {train_data.shape}")
                # Print data statistics for debugging
                state_data = train_data[:, :, u_dim:]
                print(f"  Data stats: mean={np.mean(state_data):.4f}, std={np.std(state_data):.4f}, "
                      f"min={np.min(state_data):.4f}, max={np.max(state_data):.4f}")

                # ==================== MLP ====================
                print("  Training MLP...")
                start_time = time.time()

                mlp_model = MLPDynamics(state_dim, u_dim, mlp_hidden_dim, HIDDEN_LAYERS)
                mlp_model, val_hist, val_loss = train_mlp(
                    mlp_model, train_data, val_data, u_dim, device,
                    gamma=gamma, train_steps=train_steps,
                    batch_size=batch_size, num_workers=num_workers,
                    use_compile=use_compile
                )

                mlp_train_time = time.time() - start_time
                mlp_test_loss = evaluate_kstep_error(mlp_model, test_data, u_dim, device, gamma)
                mlp_params = mlp_model.count_parameters()

                # Use scientific notation for very small values
                mlp_loss_str = f"{mlp_test_loss:.2e}" if mlp_test_loss < 1e-4 else f"{mlp_test_loss:.6f}"
                print(f"    test_Kloss: {mlp_loss_str}, Params: {mlp_params:,}, Time: {mlp_train_time:.1f}s")

                # Save MLP model (matching train_model.py format)
                mlp_path = os.path.join(model_dir, f"mlp_{config_key}.pt")
                torch.save({
                    'model': mlp_model.state_dict(),  # 'model' key like train_model.py
                    'layer': [state_dim + u_dim] + [mlp_hidden_dim] * HIDDEN_LAYERS + [state_dim],  # Architecture
                    'state_dim': state_dim,
                    'u_dim': u_dim,
                    'env_name': env_name,
                    'encode_dim_param': encode_dim_param,
                    'hidden_dim': mlp_hidden_dim,
                    'hidden_layers': HIDDEN_LAYERS,
                    'train_samples': train_samples,
                    'test_Kloss': mlp_test_loss,  # Matching train_model.py column name
                    'best_val_Kloss': val_loss,
                    'val_history': val_hist,
                    'seed': seed,
                    'gamma': gamma,
                    'train_steps': train_steps,
                }, mlp_path)
                print(f"    Saved: {mlp_path}")

                # Record MLP result (matching train_model.py CSV format)
                results.append({
                    'env_name': env_name,
                    'state_dim': state_dim,
                    'u_dim': u_dim,
                    'method': 'MLP',
                    'train_samples': train_samples,
                    'encode_dim_param': encode_dim_param,
                    'encode_dim': actual_encode_dim,
                    'seed': seed,
                    'gamma': gamma,
                    'hidden_layers': HIDDEN_LAYERS,
                    'hidden_dim': mlp_hidden_dim,
                    'test_Kloss': mlp_test_loss,
                    'best_val_Kloss': val_loss,
                    'num_params': mlp_params,
                    'train_time': mlp_train_time,
                    'model_path': mlp_path,
                })

                # ==================== EDMD ====================
                print("  Training EDMD...")
                start_time = time.time()

                print(f"    Using hierarchical lifting: target_dim={state_dim * (1 + encode_dim_param)}")

                try:
                    edmd_model = train_edmd(train_data, u_dim, encode_dim_param=encode_dim_param)
                    edmd_train_time = time.time() - start_time
                    edmd_test_loss = evaluate_kstep_error(edmd_model, test_data, u_dim, device, gamma)
                    edmd_params = edmd_model.count_parameters()

                    # Use scientific notation for very small values
                    edmd_loss_str = f"{edmd_test_loss:.2e}" if edmd_test_loss < 1e-4 else f"{edmd_test_loss:.6f}"
                    print(f"    test_Kloss: {edmd_loss_str}, Lift dim: {edmd_model.lift_dim}, "
                          f"Max degree: {edmd_model.max_degree}, #Features: {len(edmd_model.feature_specs)}, "
                          f"Time: {edmd_train_time:.1f}s")

                    # Save EDMD model (as .pt for consistency, using torch for the matrices)
                    edmd_path = os.path.join(model_dir, f"edmd_{config_key}.pt")
                    torch.save({
                        'A': torch.from_numpy(edmd_model.A),
                        'B': torch.from_numpy(edmd_model.B),
                        'state_dim': edmd_model.state_dim,
                        'u_dim': edmd_model.u_dim,
                        'env_name': env_name,
                        'encode_dim_param': encode_dim_param,
                        'max_degree': edmd_model.max_degree,
                        'lift_dim': edmd_model.lift_dim,
                        'target_dim': edmd_model.target_dim,
                        'n_features': len(edmd_model.feature_specs),
                        'feature_specs': edmd_model.feature_specs,  # For complete reconstruction
                        'train_samples': train_samples,
                        'test_Kloss': edmd_test_loss,  # Matching train_model.py column name
                        'seed': seed,
                        'gamma': gamma,
                    }, edmd_path)

                    # Also save as .npz for numpy compatibility
                    edmd_npz_path = os.path.join(model_dir, f"edmd_{config_key}.npz")
                    edmd_model.save(edmd_npz_path)
                    print(f"    Saved: {edmd_path}")

                except Exception as e:
                    print(f"    EDMD failed: {e}")
                    edmd_test_loss = np.nan
                    edmd_train_time = 0
                    edmd_params = 0
                    edmd_path = ""

                # Record EDMD result (matching train_model.py CSV format)
                results.append({
                    'env_name': env_name,
                    'state_dim': state_dim,
                    'u_dim': u_dim,
                    'method': 'EDMD',
                    'train_samples': train_samples,
                    'encode_dim_param': encode_dim_param,
                    'encode_dim': actual_encode_dim,
                    'seed': seed,
                    'gamma': gamma,
                    'test_Kloss': edmd_test_loss,
                    'best_val_Kloss': np.nan,  # EDMD doesn't use validation
                    'num_params': edmd_params,
                    'train_time': edmd_train_time,
                    'model_path': edmd_path,
                })

                # Load Koopman result for comparison
                koopman_loss = load_koopman_results(env_name, train_samples, koopman_log_dir)
                if not np.isnan(koopman_loss):
                    # Use scientific notation for very small values
                    if koopman_loss < 1e-4:
                        print(f"    Koopman test_Kloss (from log): {koopman_loss:.2e}")
                    else:
                        print(f"    Koopman test_Kloss (from log): {koopman_loss:.6f}")

                # Save results incrementally
                pd.DataFrame(results).to_csv(results_csv_path, index=False)
                print(f"    Results updated: {results_csv_path}")

    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"\nSaved files:")
    print(f"  Results CSV: {results_csv_path}")
    print(f"  Model checkpoints: {model_dir}/")
    print(f"    - MLP models: mlp_{{env}}_{{seed}}_{{encode_dim_param}}_{{train_samples}}.pt")
    print(f"    - EDMD models: edmd_{{env}}_{{seed}}_{{encode_dim_param}}_{{train_samples}}.pt/.npz")
    print('='*70)

    # Generate plots and summary (use log_base_dir for all outputs)
    results_df = pd.DataFrame(results)
    generate_summary_and_plots(results_df, log_base_dir, koopman_log_dir)

    return results_df


def generate_summary_and_plots(results_df: pd.DataFrame, output_dir: str, koopman_log_dir: str):
    """Generate summary statistics and paper-quality plots."""

    if results_df.empty:
        print("No results to plot")
        return

    print("\n" + "="*70)
    print("GENERATING SUMMARY AND PLOTS")
    print("="*70)

    # Add Koopman results for comparison
    koopman_results = []
    for _, row in results_df[results_df['method'] == 'MLP'].drop_duplicates(
        subset=['env_name', 'train_samples']).iterrows():
        koopman_loss = load_koopman_results(row['env_name'], row['train_samples'], koopman_log_dir)
        koopman_results.append({
            'env_name': row['env_name'],
            'train_samples': row['train_samples'],
            'Koopman_Kloss': koopman_loss
        })
    koopman_df = pd.DataFrame(koopman_results)

    # ==================== Summary Table ====================
    summary = results_df.groupby(['env_name', 'method', 'train_samples']).agg({
        'test_Kloss': ['mean', 'std', 'min'],
        'train_time': 'mean',
        'num_params': 'first'
    }).round(6)

    summary_path = os.path.join(output_dir, "baseline_summary.csv")
    summary.to_csv(summary_path)
    print(f"\nSummary saved to: {summary_path}")
    print(summary.to_string())

    # ==================== Plot 1: Bar chart by environment ====================
    max_samples = results_df['train_samples'].max()
    max_encode_param = results_df['encode_dim_param'].max()

    plot_df = results_df[
        (results_df['train_samples'] == max_samples) &
        (results_df['encode_dim_param'] == max_encode_param)
    ]

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))

        envs = plot_df['env_name'].unique()
        methods = ['EDMD', 'MLP']
        x = np.arange(len(envs))
        width = 0.25
        colors = {'EDMD': '#1f77b4', 'MLP': '#ff7f0e', 'Koopman': '#2ca02c'}

        for i, method in enumerate(methods):
            means = []
            stds = []
            for env in envs:
                env_method = plot_df[(plot_df['env_name'] == env) &
                                    (plot_df['method'] == method)]
                means.append(env_method['test_Kloss'].mean())
                stds.append(env_method['test_Kloss'].std())

            offset = (i - 1) * width
            ax.bar(x + offset, means, width, yerr=stds,
                  label=method, color=colors[method], capsize=3, alpha=0.8)

        # Add Koopman bars
        koopman_means = []
        for env in envs:
            k_row = koopman_df[(koopman_df['env_name'] == env) &
                              (koopman_df['train_samples'] == max_samples)]
            if not k_row.empty:
                koopman_means.append(k_row['Koopman_Kloss'].values[0])
            else:
                koopman_means.append(np.nan)

        ax.bar(x + width, koopman_means, width,
              label='Koopman (Ours)', color=colors['Koopman'], alpha=0.8)

        ax.set_xlabel('Environment')
        ax.set_ylabel('Prediction Error (test_Kloss)')
        ax.set_title(f'Baseline Comparison (train_samples={max_samples:,}, encode_dim_param={max_encode_param})')
        ax.set_xticks(x)
        ax.set_xticklabels(envs)
        ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', alpha=0.5, axis='y')

        plt.tight_layout()

        for fmt in ['pdf', 'png']:
            path = os.path.join(output_dir, f'baseline_comparison.{fmt}')
            fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved: baseline_comparison.pdf/png")
        plt.close(fig)

    # ==================== Plot 2: Scaling with train samples ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, env in enumerate(ENVS[:4]):
        ax = axes[idx]

        for method in ['MLP', 'EDMD']:
            method_df = results_df[(results_df['env_name'] == env) &
                                  (results_df['method'] == method)]

            if method_df.empty:
                continue

            agg = method_df.groupby('train_samples').agg({
                'test_Kloss': ['mean', 'std']
            }).reset_index()
            agg.columns = ['train_samples', 'mean', 'std']

            ax.errorbar(agg['train_samples'], agg['mean'], yerr=agg['std'],
                       marker='o', label=method, capsize=3)

        # Add Koopman line
        env_koopman = koopman_df[koopman_df['env_name'] == env]
        if not env_koopman.empty:
            ax.plot(env_koopman['train_samples'], env_koopman['Koopman_Kloss'],
                   'g^-', label='Koopman (Ours)', markersize=8)

        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Prediction Error (test_Kloss)')
        ax.set_title(env)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'baseline_scaling_samples.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: baseline_scaling_samples.pdf/png")
    plt.close(fig)

    # ==================== Plot 3: Scaling with encode_dim_param ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, env in enumerate(ENVS[:4]):
        ax = axes[idx]

        for method in ['MLP', 'EDMD']:
            method_df = results_df[(results_df['env_name'] == env) &
                                  (results_df['method'] == method) &
                                  (results_df['train_samples'] == max_samples)]

            if method_df.empty:
                continue

            agg = method_df.groupby('encode_dim_param').agg({
                'test_Kloss': ['mean', 'std']
            }).reset_index()
            agg.columns = ['encode_dim_param', 'mean', 'std']

            color = '#ff7f0e' if method == 'MLP' else '#1f77b4'
            ax.errorbar(agg['encode_dim_param'], agg['mean'], yerr=agg['std'],
                       marker='o', label=method, capsize=3, color=color)

        # Add Koopman horizontal line
        k_row = koopman_df[(koopman_df['env_name'] == env) &
                          (koopman_df['train_samples'] == max_samples)]
        if not k_row.empty:
            ax.axhline(y=k_row['Koopman_Kloss'].values[0],
                      color='green', linestyle='--', label='Koopman (Ours)')

        ax.set_xlabel('Encode Dim Param (multiplier)')
        ax.set_ylabel('Prediction Error (test_Kloss)')
        ax.set_title(env)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        path = os.path.join(output_dir, f'baseline_scaling_encode_dim.{fmt}')
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved: baseline_scaling_encode_dim.pdf/png")
    plt.close(fig)

    # ==================== Improvement Table ====================
    print("\n" + "-"*70)
    print("IMPROVEMENT OVER BASELINES (at max train_samples)")
    print("-"*70)

    for env in ENVS:
        k_loss = koopman_df[(koopman_df['env_name'] == env) &
                          (koopman_df['train_samples'] == max_samples)]['Koopman_Kloss'].values

        mlp_loss = results_df[(results_df['env_name'] == env) &
                            (results_df['method'] == 'MLP') &
                            (results_df['train_samples'] == max_samples) &
                            (results_df['encode_dim_param'] == max_encode_param)]['test_Kloss'].mean()

        edmd_loss = results_df[(results_df['env_name'] == env) &
                             (results_df['method'] == 'EDMD') &
                             (results_df['train_samples'] == max_samples) &
                             (results_df['encode_dim_param'] == max_encode_param)]['test_Kloss'].mean()

        if len(k_loss) > 0 and not np.isnan(k_loss[0]):
            k = k_loss[0]
            mlp_impr = (mlp_loss - k) / mlp_loss * 100 if not np.isnan(mlp_loss) else np.nan
            edmd_impr = (edmd_loss - k) / edmd_loss * 100 if not np.isnan(edmd_loss) else np.nan
            print(f"{env:20s}: Koopman={k:.6f}, MLP={mlp_loss:.6f} ({mlp_impr:+.1f}%), "
                  f"EDMD={edmd_loss:.6f} ({edmd_impr:+.1f}%)")
        else:
            print(f"{env:20s}: MLP={mlp_loss:.6f}, EDMD={edmd_loss:.6f}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline comparison for Koopman paper (MLP and EDMD)")
    parser.add_argument('--output_dir', type=str, default='BaselineResults_Full_v2',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu')
    parser.add_argument('--train_steps', type=int, default=TRAIN_STEPS,
                       help='Total training iterations for MLP (matching Koopman)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip existing results (default: overwrite all)')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data when real data unavailable')
    # Speed optimization options
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: reduced train_steps (10k) for quick testing')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE}, use 256+ for faster GPU)')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                       help=f'DataLoader workers (default: {NUM_WORKERS}, 0 for debugging)')
    parser.add_argument('--compile', action='store_true',
                       help='Use torch.compile() for PyTorch 2.0+ speedup')
    parser.add_argument('--envs', type=str, nargs='+', default=None,
                       help='Specific environments to run (default: all)')

    args = parser.parse_args()

    # Fast mode overrides
    if args.fast:
        args.train_steps = TRAIN_STEPS_FAST
        print(f"Fast mode: train_steps={args.train_steps}")

    output_dir = os.path.join(os.path.dirname(__file__), args.output_dir)

    # Override ENVS if specified (modify module-level variable)
    envs_to_run = args.envs if args.envs else ENVS
    if args.envs:
        print(f"Running on environments: {envs_to_run}")

    run_experiment(
        output_dir=output_dir,
        device_str=args.device,
        train_steps=args.train_steps,
        skip_existing=args.skip_existing,
        use_synthetic=args.synthetic,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_compile=args.compile,
        envs=envs_to_run
    )
