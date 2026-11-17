"""Train and export the small damped-pendulum Koopman model used by the project page (docs/).

Steps:
1. Train a neural Koopman model with ``train_model.train`` on swings within +/-90 deg under random
   torques (the linear-input model cannot represent the cos(theta) * u actuation over a full turn).
2. Keep the learned encoder and re-estimate (A, B) by least squares (EDMD) on fresh, independent
   transitions -- the fixed-feature, sample-split setting analysed in the paper.
3. Report forecast errors under the page's torque scheme and design a 15-step MPC on the learned model.
4. Write ``docs/assets/demo_model.json`` (encoder weights, A, B, MPC feedback gains, error statistics).

Usage (from scripts/):
    python export_page_demo.py                    # train, then export the demo model
    python export_page_demo.py --checkpoint PATH  # export from an existing checkpoint
    python export_page_demo.py --grid             # train/export the data-size x latent-size grid

With --grid, one model is trained per (m, n_mult) and (A, B) are re-estimated by EDMD on that model's
own m training windows, so every model uses exactly its data budget. Writes docs/assets/grid_models.json.
"""
import argparse
import glob
import json
import os
import random
import sys

import numpy as np
import torch
from scipy.integrate import odeint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
import dataset as D  # noqa: E402
from network import KoopmanNet  # noqa: E402

G, B_DAMP, DT, UMAX = 9.8, 1.0, 0.02, 8.0
PROJECT = "page_demo"
# the +/-90 deg, 16k-trajectory dataset that train_model() writes (other pendulum caches may exist)
DATASET = os.path.join(HERE, "..", "data", "datasets",
                       "dataset_DampingPendulum_v2_nonorm_Ktrain_16000_Kval_2000_Ktest_2000_Ksteps_15.pt")


def pendulum_step(x, u):
    f = lambda y, t: [y[1], -G * np.sin(y[0]) - B_DAMP * y[1] + np.cos(y[0]) * u]
    return odeint(f, x, [0.0, DT])[-1]


def train_model(project=PROJECT, train_samples=16000, encode_dim=8, train_steps=20000):
    D.DampingPendulumDataCollector.random_state = lambda self: np.array(
        [random.uniform(-np.pi / 2, np.pi / 2), random.uniform(-3.0, 3.0)])
    import train_model as tm
    tm.train(project_name=project, env_name="DampingPendulum", max_train_samples=16000, train_samples=train_samples,
             val_samples=2000, test_samples=2000, Ksteps=15, train_steps=train_steps, encode_dim=encode_dim,
             hidden_layers=2, hidden_dim=64, gamma=0.8, seed=17382, batch_size=64, val_step=1000, lr_step=1000,
             lr_gamma=0.9, max_norm=0.1, use_control_loss=True, use_covariance_loss=True,
             multiply_encode_by_input_dim=True)
    return sorted(glob.glob(os.path.join(HERE, "..", "log", project, "best_models", "*.pth")),
                  key=os.path.getmtime)[-1]


def load_net(path):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    layers, sd = ck["layer"], ck["model"]
    net = KoopmanNet(layers, 2 + layers[-1], 1, use_residual=True)
    net.load_state_dict(sd)
    net.eval()
    return net, sd, 2 + layers[-1]


def encoder_blocks(sd, digits=6):
    rnd = lambda a: np.round(np.asarray(a, float), digits).tolist()
    sdn = {k: v.numpy() for k, v in sd.items()}
    blocks, i = [], 0
    while f"encode_net.{i}.linear.weight" in sdn:
        blk = {"W": rnd(sdn[f"encode_net.{i}.linear.weight"]), "b": rnd(sdn[f"encode_net.{i}.linear.bias"])}
        if f"encode_net.{i}.residual.weight" in sdn:
            blk["Wr"] = rnd(sdn[f"encode_net.{i}.residual.weight"])
            blk["br"] = rnd(sdn[f"encode_net.{i}.residual.bias"])
        blocks.append(blk)
        i += 1
    return blocks


def forecast_errors(enc, A, B, seed=7, swings=300):
    rng = np.random.default_rng(seed)
    horizons = {15: [], 50: [], 100: []}
    for _ in range(swings):
        x = np.array([rng.uniform(-1.3, 1.3), rng.uniform(-2, 2)])
        z, u = enc(x)[0], 0.0
        for k in range(100):
            u = page_torque(rng, k, u)
            x = pendulum_step(x, u)
            z = A @ z + B * u
            if k + 1 in horizons:
                horizons[k + 1].append(np.degrees(abs(z[0] - x[0])))
    return {f"{k * DT:.1f}s": [round(float(np.median(v)), 2), round(float(np.quantile(v, 0.9)), 2)]
            for k, v in horizons.items()}


def run_grid(out_path, data_sizes=(2, 5, 20, 60, 250, 1000, 16000), mults=(1, 4, 16), train_steps=15000):
    models = []
    for m in data_sizes:
        for mult in mults:
            project = f"page_grid_m{m}_k{mult}"
            done = glob.glob(os.path.join(HERE, "..", "log", project, "best_models", "*.pth"))
            path = (max(done, key=os.path.getmtime) if done else
                    train_model(project=project, train_samples=m, encode_dim=mult, train_steps=train_steps))
            net, sd, n = load_net(path)
            enc = lambda X: net.encode(torch.tensor(np.atleast_2d(X), dtype=torch.float32)).detach().numpy().astype(float)
            data = torch.load(DATASET,
                              weights_only=False)
            tr = np.asarray(data["Ktrain_data"])[:, :m]          # the same m windows the encoder was trained on
            Xu, Xn = tr[:-1].reshape(-1, 3), tr[1:].reshape(-1, 3)
            M = np.linalg.lstsq(np.hstack([enc(Xu[:, 1:]), Xu[:, :1]]), enc(Xn[:, 1:]), rcond=None)[0].T
            A, B = M[:, :n], M[:, n]
            stats = forecast_errors(enc, A, B)
            print(f"m={m:6d} n={n:3d}: forecast median error 0.3s/1s/2s =",
                  stats["0.3s"][0], stats["1.0s"][0], stats["2.0s"][0])
            rnd = lambda a: np.round(np.asarray(a, float), 5).tolist()
            models.append({"m": m, "mult": mult, "n": n, "blocks": encoder_blocks(sd, 5), "A": rnd(A), "B": rnd(B),
                           "pred_err_deg": stats})
    with open(out_path, "w") as fh:
        json.dump({"dt": DT, "umax": UMAX, "note": "Damped-pendulum Koopman models trained with train_model.py; "
                   "A, B re-estimated by EDMD on each model's own training windows.", "models": models},
                  fh, separators=(",", ":"))
    print("wrote", os.path.relpath(out_path))


def page_torque(rng, k, u):
    """Piecewise-constant random torque, as in the page demo (new value every 10 steps)."""
    return rng.uniform(-UMAX, UMAX) * 0.8 if k % 10 == 0 else u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "docs", "assets", "demo_model.json"))
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()
    if args.grid:
        run_grid(os.path.join(HERE, "..", "docs", "assets", "grid_models.json"))
        return
    ckpt_path = args.checkpoint or train_model()
    net, sd, n = load_net(ckpt_path)
    enc = lambda X: net.encode(torch.tensor(np.atleast_2d(X), dtype=torch.float32)).detach().numpy().astype(float)

    # EDMD re-estimate of (A, B) on fresh transitions
    rng = np.random.default_rng(1)
    X, U, Y = [], [], []
    for _ in range(4000):
        x = np.array([rng.uniform(-np.pi / 2, np.pi / 2), rng.uniform(-3, 3)])
        for _ in range(15):
            u = rng.uniform(-UMAX, UMAX)
            xn = pendulum_step(x, u)
            X.append(x); U.append(u); Y.append(xn)
            x = xn
    X, U, Y = map(np.array, (X, U, Y))
    M = np.linalg.lstsq(np.hstack([enc(X), U[:, None]]), enc(Y), rcond=None)[0].T
    A, B = M[:, :n], M[:, n]

    # forecast errors on held-out swings, page torque scheme
    stats = forecast_errors(enc, A, B)
    rng = np.random.default_rng(7)
    print("forecast |angle error| deg (median, 90%):", stats)

    # 15-step MPC on the learned model; its first input is a linear feedback u = -Kz z + Kr [theta_ref, 0]
    H, q_theta, q_omega, r = 15, 50.0, 1.0, 0.01
    Sx = np.vstack([np.linalg.matrix_power(A, k) for k in range(1, H + 1)])
    Su = np.zeros((H * n, H))
    for k in range(1, H + 1):
        for j in range(k):
            Su[(k - 1) * n:k * n, j] = np.linalg.matrix_power(A, k - 1 - j) @ B
    W = np.zeros(H * n); W[0::n] = q_theta; W[1::n] = q_omega
    gain = np.linalg.solve(Su.T @ (W[:, None] * Su) + r * np.eye(H), Su.T * W)[0]
    Kz = gain @ Sx
    T = np.zeros((H * n, 2)); T[0::n, 0] = 1; T[1::n, 1] = 1
    Kr = gain @ T

    wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi
    final = []
    for target_deg in (-30, -15, 0, 15, 30):
        target = np.radians(target_deg)
        for _ in range(10):
            x = np.array([rng.uniform(-1.4, 1.4), rng.uniform(-2, 2)])
            for _ in range(300):
                x = pendulum_step(x, float(np.clip(-Kz @ enc(x)[0] + Kr @ [target, 0.0], -UMAX, UMAX)))
            final.append(abs(np.degrees(wrap(x[0] - target))))
    print(f"MPC final angle error: median {np.median(final):.2f} deg, max {np.max(final):.2f} deg")

    rnd = lambda a: np.round(np.asarray(a, float), 6).tolist()
    blocks = encoder_blocks(sd)
    out = {"note": "Damped-pendulum neural Koopman model (scripts/export_page_demo.py): encoder trained with "
                   "train_model.py on |theta0|<=90deg; A, B re-estimated by EDMD on fresh data.",
           "dt": DT, "umax": UMAX, "blocks": blocks, "A": rnd(A), "B": rnd(B), "Kz": rnd(Kz), "Kr": rnd(Kr),
           "pred_err_deg": stats, "mpc_final_err_deg": [round(float(np.median(final)), 2), round(float(np.max(final)), 2)]}
    with open(args.out, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print("wrote", os.path.relpath(args.out))


if __name__ == "__main__":
    main()
