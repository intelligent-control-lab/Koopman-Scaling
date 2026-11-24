"""Diagnostics of the Kinova auxiliary-loss study (kinova_aux_sweep.py), on held-out real-robot data.

For every run Kinaux__<arm>_<weights>__m<m>_s<seed>: held-out input-reconstruction error of the inverse-control map
u_hat = (B^T B)^+ B^T (z+ - A z) (as in regularizer_diagnostics.py), normalized leverage kappa_hat/n and Gram condition
number on 60,000 test states, and the logged validation / test prediction losses (normalized data).
Usage: python kinova_diagnostics.py DATASET.npz OUT.csv RUN_DIR [RUN_DIR ...]
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
from network import KoopmanNet  # noqa: E402

U_DIM = 7
ds, out, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
te = np.load(ds)["Ktest_data"]                              # (16, 20000, 7 + 14), normalized
rng = np.random.default_rng(0)
W = te[:, rng.choice(te.shape[1], 4000, replace=False)]
x, u, xn = (W[:-1, :, U_DIM:].reshape(-1, 14), W[:-1, :, :U_DIM].reshape(-1, U_DIM), W[1:, :, U_DIM:].reshape(-1, 14))
S = te[:, :, U_DIM:].reshape(-1, 14)
S = torch.tensor(S[rng.choice(len(S), 60000, replace=False)], dtype=torch.double)
h = len(S) // 2
rows = []
for d in sorted(p for r in dirs for p in glob.glob(os.path.join(r, "Kinaux__*"))):
    g = re.search(r"Kinaux__(\w+?)_(w\S+?|c\S+?_u\S+?)__m(\d+)_s(\d+)$", d)
    ck = sorted(glob.glob(os.path.join(d, "best_models", "*.pth")))
    if not g or not ck:
        continue
    log = pd.read_csv(os.path.join(d, "koopman_results_log.csv")).iloc[-1]
    c = torch.load(ck[-1], weights_only=False, map_location="cpu")
    net = KoopmanNet(c["layer"], c["layer"][0] + c["layer"][-1], c["model"]["lB.weight"].shape[1], use_residual=True).double()
    net.load_state_dict(c["model"])
    A, B = net.lA.weight.data.numpy(), net.lB.weight.data.numpy()
    with torch.no_grad():
        z = net.encode(torch.tensor(x, dtype=torch.double)).numpy()
        zn = net.encode(torch.tensor(xn, dtype=torch.double)).numpy()
        P = net.encode(S).numpy()
    u_hat = (zn - z @ A.T) @ np.linalg.pinv(B).T
    G = P[:h].T @ P[:h] / h
    lam, V = np.linalg.eigh(G)
    keep = lam > 1e-12 * lam.max()
    kap = float(np.max(np.sum((P[h:] @ (V[:, keep] / np.sqrt(lam[keep]))) ** 2, axis=1)))
    w = g.group(2)
    wc, wu = (float(w[1:]), float(w[1:])) if w.startswith("w") and w != "wNone" else (
        (0.0, 0.0) if w == "wNone" else tuple(float(v) for v in re.match(r"c(\S+)_u(\S+)", w).groups()))
    if g.group(1) == "cov":
        wu = 0.0
    if g.group(1) == "ctrl":
        wc = 0.0
    rows.append(dict(arm=g.group(1), w_cov=wc, w_ctrl=wu, m=int(g.group(3)), seed=int(g.group(4)), n=P.shape[1],
                     input_rec=float(np.mean((u_hat - u) ** 2) / np.mean(u ** 2)), kappa_over_n=kap / P.shape[1],
                     cond=float(lam.max() / lam[keep].min()), val_Kloss=float(log.best_val_Kloss),
                     test_Kloss=float(log.test_Kloss)))
    print(os.path.basename(d), {k: round(v, 5) for k, v in rows[-1].items() if isinstance(v, float)}, flush=True)
pd.DataFrame(rows).drop_duplicates(["arm", "w_cov", "w_ctrl", "m", "seed"]).to_csv(out, index=False)
