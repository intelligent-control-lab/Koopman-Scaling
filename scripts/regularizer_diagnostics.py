"""Diagnostics of the four training variants (baseline, +Cov, +Ctrl, +Aux) of the tuned G1/Go2 study, at n_mult=16:
normalized leverage kappa_hat/n, Gram condition number, held-out input-reconstruction error of the inverse-control map
u_hat = (B^T B)^+ B^T (z+ - A z), test prediction loss, and tracking survival.
Usage: python regularizer_diagnostics.py ENV DATASET RUNS_GLOB FINAL_DIR OUT.csv"""
import glob, os, re, sys
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
from network import KoopmanNet  # noqa: E402
U = {"G1": 23, "Go2": 12}
env, ds, runs_glob, final_dir, out = sys.argv[1:6]
te = np.load(ds)["Ktest_data"] if ds.endswith(".npz") else np.asarray(torch.load(ds, weights_only=False, map_location="cpu")["Ktest_data"])
u_dim = U[env]; rng = np.random.default_rng(0)
idx = rng.choice(te.shape[1], 4000, replace=False)
W = te[:, idx]                                            # (16, 4000, u+x), normalized
x, u, xn = W[:-1, :, u_dim:].reshape(-1, W.shape[-1] - u_dim), W[:-1, :, :u_dim].reshape(-1, u_dim), W[1:, :, u_dim:].reshape(-1, W.shape[-1] - u_dim)
from measure_leverage_robot import states   # same 60,000 held-out states as the leverage table
S = states(ds, u_dim)
h = len(S) // 2
tests = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(final_dir, "final_test_*.csv"))])
models = pd.concat([pd.read_csv(f, index_col=False) for f in glob.glob(os.path.join(final_dir, "final_models_*.csv"))])
rows = []
for d in sorted(glob.glob(runs_glob)):
    g = re.search(rf"aux_final_{env}_(\w+?)__m(\d+)_nm(\d+)_s(\d+)", d)
    if not g or g.group(3) != "16" or g.group(2) not in ("4000", "140000"):
        continue
    ck = sorted(glob.glob(os.path.join(d, "best_models", "*.pth")))
    if not ck:
        continue
    c = torch.load(ck[-1], weights_only=False, map_location="cpu")
    net = KoopmanNet(c["layer"], c["layer"][0] + c["layer"][-1], c["model"]["lB.weight"].shape[1], use_residual=True).double()
    net.load_state_dict(c["model"])
    A, B = net.lA.weight.data.numpy(), net.lB.weight.data.numpy()
    with torch.no_grad():
        z, zn = net.encode(torch.tensor(x, dtype=torch.double)).numpy(), net.encode(torch.tensor(xn, dtype=torch.double)).numpy()
        P = net.encode(S).numpy()
    u_hat = (zn - z @ A.T) @ np.linalg.pinv(B).T
    rec = float(np.mean((u_hat - u) ** 2) / np.mean(u ** 2))
    G = P[:h].T @ P[:h] / h
    lam, V = np.linalg.eigh(G); keep = lam > 1e-12 * lam.max()
    kap = float(np.max(np.sum((P[h:] @ (V[:, keep] / np.sqrt(lam[keep]))) ** 2, axis=1)))
    surv = tests[tests.model_path.str.contains(os.path.basename(d) + "/")].mean_survival_steps
    cov, ctrl = {"base": (0, 0), "cov": (1, 0), "ctrl": (0, 1), "both": (1, 1)}[g.group(1)]
    mm = models[(models.train_samples == int(g.group(2))) & (models.encode_dim_param == 16) & (models.seed == int(g.group(4)))
                & (models.use_covariance_loss.astype(bool) == bool(cov)) & (models.use_control_loss.astype(bool) == bool(ctrl))]
    rows.append(dict(env=env, arm=g.group(1), m=int(g.group(2)), seed=int(g.group(4)), n=P.shape[1], kappa_over_n=kap / P.shape[1],
                     cond=float(lam.max() / lam[keep].min()), input_rec=rec,
                     test_Kloss=float(mm.test_Kloss.iloc[0]) if len(mm) else np.nan,
                     survival=float(surv.iloc[0]) if len(surv) else np.nan))
    print(rows[-1], flush=True)
pd.DataFrame(rows).drop_duplicates(["arm", "m", "seed"]).to_csv(out, index=False)
