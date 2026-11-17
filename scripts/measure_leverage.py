"""Estimate the leverage constant kappa_n = sup_s Phi(s)^T G_n^{-1} Phi(s) of the controlled-study models.

Uses the models of ``split_study.py`` (seed 17382: log/page_grid_m*_k*, other seeds: log/split_s<seed>_m*_k*).
The pooled training states sample mu; the Gram matrix of Phi = [x; Psi(x)] is computed on one half of
them and the maximum leverage on the other half (a finite-sample lower bound on the supremum).
Writes log/leverage.csv and prints the median over seeds of kappa_n / n and the range of cond(G_n).

Usage (from scripts/):  python measure_leverage.py
"""
import csv
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
from export_page_demo import load_net  # noqa: E402
from split_study import SEEDS, SIZES, MULTS  # noqa: E402

LOG = os.path.join(HERE, "..", "log")


def main():
    data = torch.load(glob.glob(os.path.join(HERE, "..", "data", "datasets", "*DampingPendulum*.pt"))[0],
                      weights_only=False)
    X = np.asarray(data["Ktrain_data"]).reshape(-1, 3)[:, 1:]       # states (theta, omega)
    X = X[np.random.default_rng(0).permutation(len(X))]
    Xa, Xb = X[: len(X) // 2], X[len(X) // 2:]
    rows = []
    for seed in SEEDS:
        for m in SIZES:
            for mult in MULTS:
                project = f"page_grid_m{m}_k{mult}" if seed == SEEDS[0] else f"split_s{seed}_m{m}_k{mult}"
                found = glob.glob(os.path.join(LOG, project, "best_models", "*.pth"))
                if not found:
                    continue
                net, _, n = load_net(max(found, key=os.path.getmtime))
                enc = lambda S: net.encode(torch.tensor(S, dtype=torch.float32)).detach().numpy().astype(np.float64)
                Fa, Fb = enc(Xa), enc(Xb)
                G = Fa.T @ Fa / len(Fa)
                kappa = float(np.einsum("ij,ij->i", Fb @ np.linalg.pinv(G), Fb).max())
                rows.append({"seed": seed, "m": m, "n": n, "kappa": kappa, "cond_G": float(np.linalg.cond(G))})
    with open(os.path.join(LOG, "leverage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'traj':>6} " + " ".join(f"{'n=%d' % n:>8}" for n in (4, 10, 34)) + "   (median kappa_n / n over seeds)")
    for m in SIZES:
        cells = [np.median([r["kappa"] / r["n"] for r in rows if r["m"] == m and r["n"] == n]) for n in (4, 10, 34)]
        print(f"{m:6d} " + " ".join(f"{c:8.1f}" for c in cells))
    for n in (4, 10, 34):
        c = [r["cond_G"] for r in rows if r["n"] == n]
        k = [r["kappa"] / n for r in rows if r["n"] == n]
        print(f"n={n}: cond(G) {min(c):.1e}-{max(c):.1e}; kappa/n {min(k):.1f}-{max(k):.1f}, median {np.median(k):.1f}")


if __name__ == "__main__":
    main()
