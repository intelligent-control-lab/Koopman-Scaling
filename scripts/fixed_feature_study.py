"""Fixed-feature test of Proposition 1 (sampling error) on the damped pendulum.

The setting of Proposition 1: a fixed dictionary and i.i.d. transition pairs. We freeze the encoders
trained on 4000 windows (perm[:4000], from split_study.py), and fit (A, B) by least squares on m
i.i.d. transitions drawn from a disjoint pool (perm[8000:16000]: one random transition from each of
m distinct windows). Evaluation states come from a third block (perm[4000:8000]).

For each (seed, n, m) and R random draws we record
  dG       = || G^{-1/2} G_m G^{-1/2} - I ||_2   (Prop. 1 needs dG <= 1/2)
  eps_samp = relative L2 distance of the one-step state prediction from the reference fit on the
             whole pool (120k transitions), i.e. the sampling error on the state coordinates
  eps_tot  = relative one-step state prediction error.
Also recorded: gram_rel_lam_min = lambda_min / lambda_max of the reference Gram matrix of r (threshold_law_fits.py
excludes encoders whose Gram matrix is singular to numerical precision).
The regressor is psi = [Phi(x); u]; kappa is computed for psi with the pool Gram matrix, maximised
over the evaluation states.

Usage (from scripts/):  python fixed_feature_study.py
"""
import csv
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
import export_page_demo as E  # noqa: E402
from split_study import SEEDS, permutation  # noqa: E402
MULTS = tuple(int(k) for k in os.environ.get("MULTS", "1 2 4 8 16 32").split())

M_GRID = (5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 8000)
R = 40
OUT = os.path.join(HERE, "..", "log", os.environ.get("FFS_OUT", "fixed_feature_study.csv"))
REF = os.path.join(HERE, "..", "log", "ffs_reference_pool.npz")   # independent pool (make_reference_pool.py)


def encoder(seed, mult):
    project = f"page_grid_m4000_k{mult}" if seed == SEEDS[0] else f"split_s{seed}_m4000_k{mult}"
    path = max(glob.glob(os.path.join(HERE, "..", "log", project, "best_models", "*.pth")), key=os.path.basename)
    net, _, n = E.load_net(path)

    def enc(X):
        with torch.no_grad():
            return net.encode(torch.tensor(X, dtype=torch.float32)).numpy().astype(float)
    return enc, n


def regress(enc, W):
    """W: (T+1, N, 3) windows with rows [u, theta, omega]. Returns psi (N, T, n+1), z_next (N, T, n), x_next."""
    T, N = W.shape[0] - 1, W.shape[1]
    x, u, xn = W[:-1, :, 1:], W[:-1, :, :1], W[1:, :, 1:]
    phi = enc(x.reshape(-1, 2)).reshape(T, N, -1)
    zn = enc(xn.reshape(-1, 2)).reshape(T, N, -1)
    psi = np.concatenate([phi, u], axis=-1)
    return psi.transpose(1, 0, 2), zn.transpose(1, 0, 2), xn.transpose(1, 0, 2)


def rel_err(pred, target):
    return float(np.mean(np.sqrt(np.mean((pred - target) ** 2, axis=0) / np.mean(target ** 2, axis=0))))


def rel_mse(a, b, scale):
    """Mean over state coordinates of E[(a - b)^2] / E[x+^2]."""
    return float(np.mean(np.mean((a - b) ** 2, axis=0) / scale))


def main():
    data = torch.load(E.DATASET, weights_only=False)
    W_all = np.asarray(data["Ktrain_data"])
    assert W_all.shape[1] >= 16000, W_all.shape
    rng = np.random.default_rng(0)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "mult", "n", "kappa", "gram_rel_lam_min", "m", "draw", "dG", "eps_samp", "eps_tot", "rank_def",
                    "mse_proj", "mse_samp", "mse_tot", "mse_cross"])
        Wref_all = np.load(REF)["windows"]                       # independent of the regression pool
        for seed in SEEDS:
            perm = permutation(seed, W_all.shape[1])
            pool, ev = perm[8000:16000], perm[4000:8000]
            for mult in MULTS:
                enc, n = encoder(seed, mult)
                P_psi, P_zn, _ = regress(enc, W_all[:, pool])
                R_psi, R_zn, _ = regress(enc, Wref_all)
                E_psi, _, E_xn = regress(enc, W_all[:, ev])
                Rp, Rz = R_psi.reshape(-1, n + 1), R_zn.reshape(-1, n)
                Ep, Ex = E_psi.reshape(-1, n + 1), E_xn.reshape(-1, 2)
                scale = np.mean(Ex ** 2, axis=0)
                G = Rp.T @ Rp / len(Rp)                           # reference Gram (population proxy)
                lam, V = np.linalg.eigh(G)
                rel_lam_min = float(lam.min() / lam.max())             # ~1e-16: structurally rank-deficient
                Gmh = V @ np.diag(lam ** -0.5) @ V.T
                kappa = float(np.max(np.einsum("ij,jk,ik->i", Ep, np.linalg.inv(G), Ep)))
                M_ref = np.linalg.lstsq(Rp, Rz, rcond=None)[0]
                ref_pred = (Ep @ M_ref)[:, :2]
                mse_proj = rel_mse(ref_pred, Ex, scale)
                for m in M_GRID:
                    for r in range(R):
                        win = rng.choice(len(pool), size=m, replace=False)
                        t = rng.integers(0, P_psi.shape[1], size=m)
                        X, Y = P_psi[win, t], P_zn[win, t]
                        Gm = X.T @ X / m
                        dG = float(np.linalg.norm(Gmh @ Gm @ Gmh - np.eye(n + 1), 2))
                        Mm = np.linalg.lstsq(X, Y, rcond=None)[0]
                        with np.errstate(all="ignore"):
                            pred = (Ep @ Mm)[:, :2]
                            es, et = rel_err(pred, ref_pred), rel_err(pred, Ex)
                            ms, mt = rel_mse(pred, ref_pred, scale), rel_mse(pred, Ex, scale)
                        w.writerow([seed, mult, n, kappa, rel_lam_min, m, r, dG, es, et, int(np.linalg.matrix_rank(X) < n + 1),
                                    mse_proj, ms, mt, mt - mse_proj - ms])
                print(f"seed={seed} n={n} kappa={kappa:.0f} ({kappa / (n + 1):.1f}(n+1)) proj={mse_proj:.2e}", flush=True)


if __name__ == "__main__":
    main()
