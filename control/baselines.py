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


class Planner:
    """Model-agnostic MPC planners for the planner-matched comparison.

    Every planner minimizes the objective of the closed-form linear MPC (mpc_core.MPCSolver):
        J(U) = sum_{k=0..H} ||x_k - x_ref,k||^2 + r * sum_k ||u_k||^2,
    in normalized coordinates, with inputs restricted to the normalized clamp box [lo, hi].
    The model is given by init(x_norm) -> s0 (internal state), step(s, u) -> s', and phys(s) -> x.
    mode: 'rs'   random shooting, K uniform samples (no warm start);
          'cem'  cross-entropy method, `iters` rounds of K samples, warm-started from the shifted previous plan;
          'grad' gradient descent (Adam) on one input sequence per environment, warm-started, projected onto the box.
    """

    def __init__(self, init, step, phys, H, lo, hi, r=1e-3, mode="rs", K=1000, iters=5, elite=50,
                 grad_iters=50, lr=0.1, env_chunk=50, seed=0):
        self.init, self.step, self.phys = init, step, phys
        self.H, self.lo, self.hi, self.r, self.mode = H, lo.float(), hi.float(), r, mode
        self.K, self.iters, self.elite, self.grad_iters, self.lr, self.chunk = K, iters, elite, grad_iters, lr, env_chunk
        self.gen = torch.Generator(device=lo.device).manual_seed(seed)
        self.prev = None

    def _cost(self, s0, U, ref):
        """s0: (n, D); U: (n, K, H, m); ref: (H+1, n, S). Returns (n, K)."""
        n, K = U.shape[:2]
        s = s0.unsqueeze(1).expand(n, K, s0.shape[-1]).reshape(n * K, -1)
        c = ((self.phys(s).view(n, K, -1) - ref[0].unsqueeze(1)) ** 2).sum(-1)
        for k in range(self.H):
            s = self.step(s, U[:, :, k].reshape(n * K, -1))
            c = c + ((self.phys(s).view(n, K, -1) - ref[k + 1].unsqueeze(1)) ** 2).sum(-1)
        return c + self.r * (U ** 2).sum((-1, -2))

    def __call__(self, x_norm, X_ref_norm):
        N = x_norm.shape[0]
        m = self.lo.shape[0]
        dev = x_norm.device
        if self.prev is None:
            self.prev = torch.zeros(N, self.H, m, device=dev).clamp(self.lo, self.hi)
        warm = torch.cat([self.prev[:, 1:], self.prev[:, -1:]], dim=1)
        out_plan = torch.empty_like(warm)
        for a in range(0, N, self.chunk):
            b = min(N, a + self.chunk)
            x = x_norm[a:b].float()
            ref = X_ref_norm[:, a:b].float()
            if self.mode == "grad":
                with torch.inference_mode(False), torch.enable_grad():
                    s0 = self.init(x.clone()).detach()
                    ref_g = ref.clone()
                    U = warm[a:b].clone().unsqueeze(1).requires_grad_(True)       # (n, 1, H, m)
                    opt = torch.optim.Adam([U], lr=self.lr)
                    for _ in range(self.grad_iters):
                        opt.zero_grad()
                        self._cost(s0, U, ref_g).sum().backward()
                        opt.step()
                        with torch.no_grad():
                            U.data = torch.max(torch.min(U.data, self.hi), self.lo)
                    U_best = U.detach()[:, 0].clone()
                out_plan[a:b] = U_best
                continue
            with torch.no_grad():
                s0 = self.init(x)
                n = b - a
                if self.mode == "rs":
                    U = self.lo + (self.hi - self.lo) * torch.rand(n, self.K, self.H, m, device=dev, generator=self.gen)
                    out_plan[a:b] = U[torch.arange(n, device=dev), self._cost(s0, U, ref).argmin(1)]
                    continue
                mu, sd = warm[a:b].clone(), (0.5 * (self.hi - self.lo)).expand(n, self.H, m).clone()
                best_c = torch.full((n,), float("inf"), device=dev)
                best_U = mu.clone()
                for _ in range(self.iters):
                    U = mu.unsqueeze(1) + sd.unsqueeze(1) * torch.randn(n, self.K, self.H, m, device=dev, generator=self.gen)
                    U = torch.max(torch.min(U, self.hi), self.lo)
                    U[:, 0] = mu                                                    # keep the current mean as a candidate
                    c = self._cost(s0, U, ref)
                    idx = c.topk(self.elite, dim=1, largest=False).indices          # (n, E)
                    el = U[torch.arange(n, device=dev).unsqueeze(1), idx]          # (n, E, H, m)
                    mu, sd = el.mean(1), el.std(1) + 1e-3
                    cmin, imin = c.min(1)
                    better = cmin < best_c
                    best_c = torch.where(better, cmin, best_c)
                    best_U[better] = U[torch.arange(n, device=dev), imin][better]
                out_plan[a:b] = best_U
        self.prev = out_plan
        return out_plan[:, 0].double()
