"""Baseline models for the G1/Go2 closed-loop comparison (Table 5 of the paper).

- EDMD: polynomial lifting z = [x, polynomial features of x] with z+ = A z + B u. It exposes the same
  interface as KoopmanNet (encode, lA, lB), so mpc_core.MPCSolver runs the same linear MPC on it.
- NNDM: an MLP x+ = f(x, u), controlled by random-shooting MPC: sample K input sequences uniformly in the
  normalized input box, roll them out with the MLP, and apply the first input of the cheapest sequence.
  The cost is the same as in the linear MPC: sum over k = 0..H of ||x_k - x_ref,k||^2 in normalized units.

All models work in the normalized coordinates of the training dataset (norm_stats of the dataset file).
No Isaac Lab imports, so the module can be tested alone.
"""
import numpy as np
import torch
import torch.nn as nn


class EDMDNet(nn.Module):
    """EDMD model with the KoopmanNet interface: encode(x) -> z, and lA / lB linear maps."""

    def __init__(self, A, B, feature_specs, state_dim):
        super().__init__()
        A = torch.as_tensor(np.asarray(A), dtype=torch.double)
        B = torch.as_tensor(np.asarray(B), dtype=torch.double)
        specs = np.asarray(feature_specs, dtype=np.int64)          # (L, S) exponents
        L = specs.shape[0]
        k = max(int((specs > 0).sum(1).max()), 1)                 # at most k variables per feature
        idx = np.zeros((L, k), dtype=np.int64)
        exp = np.zeros((L, k), dtype=np.int64)
        for f, row in enumerate(specs):
            nz = np.nonzero(row)[0]
            idx[f, :len(nz)] = nz
            exp[f, :len(nz)] = row[nz]
        self.register_buffer("idx", torch.as_tensor(idx))
        self.register_buffer("exp", torch.as_tensor(exp, dtype=torch.double))
        self.state_dim = state_dim
        self.lA = nn.Linear(L, L, bias=False).double()
        self.lB = nn.Linear(B.shape[1], L, bias=False).double()
        self.lA.weight.data.copy_(A)
        self.lB.weight.data.copy_(B)

    def encode(self, x):
        """Polynomial lift, identical to baseline_mpc_utils.EDMD.lift (clip x to [-10, 10], features to +-1e6)."""
        x = torch.clamp(x.double(), -10.0, 10.0)
        g = x[..., self.idx]                                       # (..., L, k)
        z = torch.pow(g, self.exp).prod(dim=-1)                    # x^0 = 1 for unused slots
        return torch.clamp(z, -1e6, 1e6)


def load_edmd(path, state_dim, device):
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        A, B, specs = d["A"], d["B"], d["feature_specs"]
    else:
        d = torch.load(path, weights_only=False, map_location="cpu")
        A, B, specs = d["A"].numpy(), d["B"].numpy(), d["feature_specs"]
    net = EDMDNet(A, B, specs, state_dim).to(device)
    net.eval()
    return net


class MLPDynamics(nn.Module):
    """Same architecture as Koopman-Scaling/control/baseline_control/baseline_mpc_utils.MLPDynamics."""

    def __init__(self, state_dim, u_dim, hidden_dim=256, hidden_layers=3):
        super().__init__()
        dims = [state_dim + u_dim] + [hidden_dim] * hidden_layers + [state_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.state_dim, self.u_dim = state_dim, u_dim

    def forward(self, x, u):
        return self.net(torch.cat([x, u], dim=-1))


def load_mlp(path, device):
    c = torch.load(path, weights_only=False, map_location="cpu")
    net = MLPDynamics(int(c["state_dim"]), int(c["u_dim"]), int(c.get("hidden_dim", 256)),
                      int(c.get("hidden_layers", 3)))
    net.load_state_dict(c["model"])
    return net.to(device).eval()


class RandomShootingMPC:
    """Random-shooting MPC for the MLP model, batched over environments and samples on the GPU."""

    def __init__(self, model, H, u_low, u_high, num_samples=1000, env_chunk=250, seed=0):
        self.model, self.H, self.K, self.chunk = model, H, num_samples, env_chunk
        self.u_low = u_low.float()                                 # (m,) normalized bounds
        self.u_high = u_high.float()
        self.gen = torch.Generator(device=u_low.device).manual_seed(seed)

    @torch.no_grad()
    def __call__(self, x_norm, X_ref_norm):
        """x_norm: (N, S); X_ref_norm: (H+1, N, S). Returns (N, m) normalized inputs (double)."""
        N, S = x_norm.shape
        m = self.u_low.shape[0]
        out = torch.empty(N, m, device=x_norm.device, dtype=torch.double)
        for a in range(0, N, self.chunk):
            b = min(N, a + self.chunk)
            n = b - a
            U = torch.rand(n, self.K, self.H, m, device=x_norm.device, generator=self.gen)
            U = self.u_low + (self.u_high - self.u_low) * U        # (n, K, H, m)
            x = x_norm[a:b].float().unsqueeze(1).expand(n, self.K, S).reshape(n * self.K, S)
            ref = X_ref_norm[:, a:b].float()                       # (H+1, n, S)
            cost = ((x.view(n, self.K, S) - ref[0].unsqueeze(1)) ** 2).sum(-1)
            for k in range(self.H):
                x = self.model(x, U[:, :, k].reshape(n * self.K, m))
                cost = cost + ((x.view(n, self.K, S) - ref[k + 1].unsqueeze(1)) ** 2).sum(-1)
            best = cost.argmin(dim=1)                              # (n,)
            out[a:b] = U[torch.arange(n, device=U.device), best, 0].double()
        return out
