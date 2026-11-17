"""Controlled data-capacity study on the damped pendulum, with five seeds and a sample-split variant.

For each data size m (trajectory windows of 15 transitions), latent multiplier, and seed:
1. train the encoder with train_model.train on m windows;
2. fit (A, B) by least squares (EDMD) on
   - "same":  the same m windows (the protocol of the paper's controlled study), and
   - "split": m disjoint windows, so the features are independent of the regression data,
     which is the setting of Proposition 1;
3. record the median absolute angle error of a 1 s open-loop forecast and the spectral radius of A.

The windows are permuted per seed (seed 17382 keeps the original order, so it reuses the
export_page_demo.py --grid models). Results go to log/split_study.csv.

Usage (from scripts/):  python split_study.py
"""
import csv
import glob
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
import dataset as D  # noqa: E402
import export_page_demo as E  # noqa: E402

SEEDS = (17382, 76849, 20965, 84902, 51194)
SIZES = (2, 5, 20, 60, 250, 1000, 4000)
MULTS = (1, 4, 16)
SPLIT_OFFSET = 8000          # split windows come from positions >= 8000 of the permuted order
OUT = os.path.join(HERE, "..", "log", "split_study.csv")


def permutation(seed, n_windows):
    return np.arange(n_windows) if seed == SEEDS[0] else np.random.default_rng(seed).permutation(n_windows)


def patch_collector(perm):
    """Make train_model.train see the training windows in the order `perm`."""
    original = D.KoopmanDatasetCollector.get_data

    def get_data(self):
        tr, va, te = original(self)
        return tr[:, perm], va, te
    D.KoopmanDatasetCollector.get_data = get_data
    return original


def edmd(enc, windows):
    Xu, Xn = windows[:-1].reshape(-1, 3), windows[1:].reshape(-1, 3)   # rows [u_t, theta_t, omega_t]
    M = np.linalg.lstsq(np.hstack([enc(Xu[:, 1:]), Xu[:, :1]]), enc(Xn[:, 1:]), rcond=None)[0].T
    return M[:, :-1], M[:, -1]


def main():
    data = torch.load(E.DATASET,
                      weights_only=False)
    all_windows = np.asarray(data["Ktrain_data"])
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            done = {(int(r["seed"]), int(r["m"]), int(r["mult"])) for r in csv.DictReader(fh)}
    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["seed", "m", "mult", "n", "transitions", "err_same_1s", "err_split_1s",
                        "rho_same", "rho_split"])
        for seed in SEEDS:
            perm = permutation(seed, all_windows.shape[1])
            for m in SIZES:
                for mult in MULTS:
                    if (seed, m, mult) in done:
                        continue
                    project = (f"page_grid_m{m}_k{mult}" if seed == SEEDS[0] else
                               f"split_s{seed}_m{m}_k{mult}")
                    found = glob.glob(os.path.join(HERE, "..", "log", project, "best_models", "*.pth"))
                    if found:
                        path = max(found, key=os.path.getmtime)
                    else:
                        original = patch_collector(perm)
                        try:
                            import train_model as tm
                            E.D.DampingPendulumDataCollector.random_state = lambda self: np.array(
                                [random.uniform(-np.pi / 2, np.pi / 2), random.uniform(-3.0, 3.0)])
                            tm.train(project_name=project, env_name="DampingPendulum", max_train_samples=16000,
                                     train_samples=m, val_samples=2000, test_samples=2000, Ksteps=15,
                                     train_steps=15000, encode_dim=mult, hidden_layers=2, hidden_dim=64,
                                     gamma=0.8, seed=seed, batch_size=64, val_step=1000, lr_step=1000,
                                     lr_gamma=0.9, max_norm=0.1, use_control_loss=True,
                                     use_covariance_loss=True, multiply_encode_by_input_dim=True)
                        finally:
                            D.KoopmanDatasetCollector.get_data = original
                        path = max(glob.glob(os.path.join(HERE, "..", "log", project, "best_models", "*.pth")),
                                   key=os.path.getmtime)
                    net, _, n = E.load_net(path)
                    enc = lambda X: net.encode(torch.tensor(np.atleast_2d(X), dtype=torch.float32)
                                               ).detach().numpy().astype(float)
                    row = [seed, m, mult, n, 15 * m]
                    errs, rhos = [], []
                    for idx in (perm[:m], perm[SPLIT_OFFSET:SPLIT_OFFSET + m]):
                        A, B = edmd(enc, all_windows[:, idx])
                        rhos.append(float(np.max(np.abs(np.linalg.eigvals(A)))))
                        with np.errstate(all="ignore"):
                            e = E.forecast_errors(enc, A, B)["1.0s"][0]
                        errs.append(e if np.isfinite(e) else float("inf"))
                    w.writerow(row + errs + rhos)
                    fh.flush()
                    print(f"seed={seed} m={m:5d} n={n:3d}: same {errs[0]:.2f} split {errs[1]:.2f} deg", flush=True)


if __name__ == "__main__":
    main()
