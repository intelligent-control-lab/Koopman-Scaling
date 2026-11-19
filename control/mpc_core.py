"""Vectorized pieces of the G1/Go2 MPC tracking evaluation (no Isaac Lab imports, so they can be tested alone).

The original loop in mpc_tracking.py evaluated the tracking metrics one environment at a time and rebuilt the
MPC prediction matrices at every control step. These functions compute exactly the same quantities:
- step_errors: the per-environment errors of one time step, for all environments at once;
- MPCSolver: the condensed MPC of Torch_MPC, with the matrices that do not depend on the state built once.
"""
import torch

# Column slices of the raw state [joint_pos, joint_vel, root_state] used by the metrics.
SLICES = {
    "G1": dict(JrPE=(0, 23), JrVE=(37, 60), acc=23, RPE=(74, 77), ROE=(77, 81), RLVE=(81, 84), RAVE=(84, 87)),
    "Go2": dict(JrPE=(0, 12), JrVE=(12, 24), acc=None, RPE=(24, 27), ROE=(27, 31), RLVE=(31, 34), RAVE=(34, 37)),
}
METRICS = ("JrPE", "JrVE", "JrAE", "RPE", "ROE", "RLVE", "RAVE")


def step_errors(name, ref_i, ref_acc_i, state_full, acc):
    """Mean absolute errors of one time step for every environment.

    ref_i: (N, S_full) reference state at step i; ref_acc_i: (N, n_joint) reference joint acceleration;
    state_full: (N, S_full) simulated state; acc: (N, n_joint_full) simulated joint acceleration.
    Returns a dict of (N,) tensors, one per metric.
    """
    s = SLICES[name]
    out = {}
    for k in ("JrPE", "JrVE", "RPE", "ROE", "RLVE", "RAVE"):
        a, b = s[k]
        out[k] = torch.abs(ref_i[:, a:b] - state_full[:, a:b]).mean(dim=1)
    acc_cols = acc if s["acc"] is None else acc[:, :s["acc"]]
    out["JrAE"] = torch.abs(ref_acc_i - acc_cols).mean(dim=1)
    return out


class MPCSolver:
    """Condensed linear MPC of Torch_MPC (mpc_tracking.py) with the state-independent matrices cached."""

    def __init__(self, net, H, m, latent_dim, s_dim, R_scale=0.0, device=None, dtype=torch.double):
        A = net.lA.weight.data
        B = net.lB.weight.data
        device = device or A.device
        Q = torch.eye(latent_dim, device=device, dtype=dtype)
        R = torch.eye(m, device=device, dtype=dtype) * R_scale
        F = torch.eye(latent_dim, device=device, dtype=dtype)
        Q[s_dim:, s_dim:] = 0.0
        F[s_dim:, s_dim:] = 0.0
        self.M = torch.cat([torch.matrix_power(A, i) for i in range(H + 1)], 0).double()
        C = torch.zeros((H + 1) * latent_dim, H * m, device=device, dtype=dtype)
        for i in range(1, H + 1):
            for j in range(i):
                C[i * latent_dim:(i + 1) * latent_dim, j * m:(j + 1) * m] = torch.matrix_power(A, i - j - 1) @ B
        self.C = C
        self.Qh = torch.block_diag(*([Q] * H + [F]))
        Rh = torch.block_diag(*([R] * H))
        P = 2 * (Rh + C.T @ self.Qh @ C)
        self.Pinv = torch.inverse(0.5 * P)
        self.net, self.H, self.m, self.latent_dim = net, H, m, latent_dim

    def __call__(self, x_k_norm, X_ref_norm):
        z_k = self.net.encode(x_k_norm).permute(1, 0)
        Z_ref = self.net.encode(X_ref_norm).permute(1, 0, 2).reshape(-1, (self.H + 1) * self.latent_dim)
        q = 2 * (z_k.T @ self.M.T - Z_ref) @ self.Qh @ self.C   # same association as Torch_MPC
        U = (-0.5 * self.Pinv @ q.T).reshape(self.H * self.m, -1)
        return U[:self.m].permute(1, 0)
