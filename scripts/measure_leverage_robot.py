"""Estimated leverage constants of the learned G1/Go2 Koopman features (compare Appendix A.4 for the pendulum).

For each saved model, encode held-out states of the training distribution (test split of the dataset, already
normalized), compute the Gram matrix G of Phi = [x; Psi(x)] on one half of the states, and take
kappa_hat = max over the other half of Phi^T G^+ Phi. Eigenvalues below 1e-12 * lambda_max are treated as zero
(numerical rank r); kappa_hat >= r in expectation. A maximum over finite samples underestimates the supremum.

Usage: python measure_leverage_robot.py ENV DATASET RUNS_GLOB OUT.csv [ARMS]
"""
import glob
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
from network import KoopmanNet  # noqa: E402

U_DIM = {"G1": 23, "Go2": 12}


def states(path, u_dim, n=60000, seed=0):
    if path.endswith(".npz"):
        te = np.load(path)["Ktest_data"]
    else:
        te = np.asarray(torch.load(path, weights_only=False, map_location="cpu")["Ktest_data"])
    X = te[:, :, u_dim:].reshape(-1, te.shape[-1] - u_dim)
    idx = np.random.default_rng(seed).choice(len(X), size=min(n, len(X)), replace=False)
    return torch.tensor(X[idx], dtype=torch.double)


def main():
    env, ds, runs_glob, out = sys.argv[1:5]
    arms = sys.argv[5].split(",") if len(sys.argv) > 5 else ["base"]
    X = states(ds, U_DIM[env])
    h = len(X) // 2
    rows = []
    for d in sorted(glob.glob(runs_glob)):
        g = re.search(rf"aux_final_{env}_(\w+?)__m(\d+)_nm(\d+)_s(\d+)", d)
        if not g or g.group(1) not in arms:
            continue
        ck = sorted(glob.glob(os.path.join(d, "best_models", "*.pth")))
        if not ck:
            continue
        c = torch.load(ck[-1], weights_only=False, map_location="cpu")
        layers = c["layer"]
        net = KoopmanNet(layers, layers[0] + layers[-1], c["model"]["lB.weight"].shape[1], use_residual=True).double()
        net.load_state_dict(c["model"])
        with torch.no_grad():
            P = net.encode(X).numpy()
        n = P.shape[1]
        G = P[:h].T @ P[:h] / h
        lam, V = np.linalg.eigh(G)
        keep = lam > 1e-12 * lam.max()
        W = V[:, keep] / np.sqrt(lam[keep])
        k = np.max(np.sum((P[h:] @ W) ** 2, axis=1))
        rows.append(dict(env=env, arm=g.group(1), m=int(g.group(2)), nm=int(g.group(3)), seed=int(g.group(4)), n=n,
                         rank=int(keep.sum()), kappa=float(k), kappa_over_n=float(k / n),
                         cond=float(lam.max() / lam[keep].min())))
        print(f"{env} {g.group(1)} m={g.group(2)} nm={g.group(3)} s={g.group(4)} n={n} rank={keep.sum()} "
              f"kappa/n={k / n:.1f}", flush=True)
    import pandas as pd
    pd.DataFrame(rows).drop_duplicates(["arm", "m", "nm", "seed"]).to_csv(out, index=False)


if __name__ == "__main__":
    main()
