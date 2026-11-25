"""
Baseline Control Module
=======================

This module provides MPC controllers for baseline dynamics models (MLP and EDMD)
to compare against Neural Koopman operators.

Controllers:
- MLPShootingMPC: MLP + Random Shooting MPC (nonlinear dynamics)
- EDMD_MPC: EDMD + Linear MPC (polynomial lifting with analytical QP)
"""

from .baseline_mpc_utils import (
    MLPDynamics,
    EDMD,
    load_mlp_from_checkpoint,
    load_edmd_from_checkpoint,
    normalize_state,
    denormalize_action,
    get_baseline_model_path,
)
from .mlp_mpc import MLPShootingMPC
from .edmd_mpc import EDMD_MPC

__all__ = [
    'MLPDynamics',
    'EDMD',
    'MLPShootingMPC',
    'EDMD_MPC',
    'load_mlp_from_checkpoint',
    'load_edmd_from_checkpoint',
    'normalize_state',
    'denormalize_action',
    'get_baseline_model_path',
]