"""Representation diagnostics for any learned feature matrix: Gram spectrum, numerical rank, condition number, and
estimated leverage (the quantity kappa_n of the paper's sampling bound).

Input: an (N_states, n) array of features Phi(s) evaluated on states drawn from the sampling distribution
(.npy, .npz with key "features", or .csv without header). The Gram matrix is estimated on one half of the rows and
the leverage Phi(s)^T G^+ Phi(s) on the other half, so that kappa_hat is not biased by the fit; eigenvalues below
RANK_TOL * lambda_max are treated as zero. kappa_hat is an empirical diagnostic, not a certified bound: it is a
maximum over finitely many states and depends on rare states.

Usage:  python leverage_diagnostics.py FEATURES.npy [--seed 0] [--rank-tol 1e-12] [--plot leverage.pdf]
"""
import argparse

import numpy as np


def load(path):
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        return np.load(path)["features"]
    return np.loadtxt(path, delimiter=",")


def diagnostics(P, seed=0, rank_tol=1e-12):
    P = np.asarray(P, dtype=np.float64)
    idx = np.random.default_rng(seed).permutation(len(P))
    fit, ev = P[idx[: len(P) // 2]], P[idx[len(P) // 2:]]
    G = fit.T @ fit / len(fit)
    lam, V = np.linalg.eigh(G)
    keep = lam > rank_tol * lam.max()
    W = V[:, keep] / np.sqrt(lam[keep])
    lev = np.sum((ev @ W) ** 2, axis=1)                     # per-state leverage Phi(s)^T G^+ Phi(s)
    n = P.shape[1]
    return dict(n=n, rank=int(keep.sum()), lam_min=float(lam.min()), lam_max=float(lam.max()),
                cond=float(lam.max() / lam[keep].min()), rel_lam_min=float(lam.min() / lam.max()),
                kappa_hat=float(lev.max()), kappa_over_n=float(lev.max() / n),
                mean_leverage=float(lev.mean()), leverage=lev)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("features")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank-tol", type=float, default=1e-12)
    ap.add_argument("--plot", default=None, help="optional output figure of the per-state leverage")
    a = ap.parse_args()
    d = diagnostics(load(a.features), a.seed, a.rank_tol)
    print(f"dimension n           {d['n']}")
    print(f"numerical rank        {d['rank']}" + ("   (rank-deficient: the full-rank assumption fails)" if d["rank"] < d["n"] else ""))
    print(f"lambda_min / max      {d['lam_min']:.3e} / {d['lam_max']:.3e}  (ratio {d['rel_lam_min']:.1e})")
    print(f"condition number      {d['cond']:.3e}  (over the numerical rank)")
    print(f"estimated leverage    {d['kappa_hat']:.1f}  (kappa_hat / n = {d['kappa_over_n']:.2f}; mean leverage = "
          f"{d['mean_leverage']:.1f}, close to the rank)")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import plot_style
        with plt.rc_context(plot_style.RC):
            fig, ax = plt.subplots(figsize=(3.2, 2.2), layout="constrained")
            ax.hist(d["leverage"] / d["n"], bins=60, color=plot_style.VARIANTS["base"]["color"])
            ax.axvline(d["kappa_over_n"], color=plot_style.VARIANTS["both"]["color"], ls="--", lw=1, label="maximum")
            ax.set(yscale="log", xlabel="leverage / n", ylabel="states")
            ax.legend()
            fig.savefig(a.plot)
        print("wrote", a.plot)


if __name__ == "__main__":
    main()
