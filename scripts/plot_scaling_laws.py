"""Figs. 2-3 of the paper (prediction error versus training samples m and latent dimension n) and the fitted exponents
of Table 2, from training logs (koopman_results_log.csv files written by train_model.py).

Configuration: runs without auxiliary losses, the five seeds, m in {1,4,16,64,140}k, n_mult in {1,2,4,8,16}.
Fit: eps = A D^-alpha + C on a log scale, pooling seeds; alpha_m is averaged over n_mult and alpha_n over m >= 16k.
95% intervals: bootstrap over seeds.

Usage (from scripts/):  python plot_scaling_laws.py --logs ../log/*/koopman_results_log.csv --out ../log/figures
"""
import argparse
import glob
import os
import sys

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

ALIASES = {"FrankaImproved": "Franka", "G1Improved": "G1", "Go2Improved": "Go2", "KinovaImproved": "Kinova"}
ORDER = [("DampingPendulum", "Damping pendulum"), ("Franka", "Franka"), ("G1", "Unitree G1"),
         ("DoublePendulum", "Double pendulum"), ("Kinova", "Kinova Gen3 (real robot)"), ("Go2", "Unitree Go2")]
SEEDS, MS, MULTS = [17382, 76849, 20965, 84902, 51194], [1000, 4000, 16000, 64000, 140000], [1, 2, 4, 8, 16]


def fit(D, E):
    D, E = np.asarray(D, float), np.asarray(E, float)
    keep = (D > 0) & (E > 0)
    D, E = D[keep], E[keep]
    if len(D) < 3:
        return None
    s, i = np.polyfit(np.log(D), np.log(E), 1)
    try:
        return curve_fit(lambda x, A, a, C: np.log(np.maximum(A * np.power(x, -a) + C, 1e-30)), D, np.log(E),
                         p0=[np.exp(i), -s, 0.0], bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]), maxfev=20000)[0]
    except Exception:
        return None


def load(paths):
    d = pd.concat([pd.read_csv(f) for f in paths], ignore_index=True)
    d["env_name"] = d.env_name.replace(ALIASES)
    d = d[(d.use_covariance_loss == False) & (d.use_control_loss == False) & d.seed.isin(SEEDS)  # noqa: E712
          & d.train_samples.isin(MS) & d.encode_dim_param.isin(MULTS)]
    d = d.drop_duplicates(subset=["env_name", "train_samples", "encode_dim_param", "seed"], keep="last")
    d["z"] = d.encode_dim + d.state_dim
    return d


def bootstrap(d, B=500, seed=0):
    """Seed-bootstrap 95% intervals for the Table 2 exponents: resample the five seeds with replacement,
    refit every curve, and average with the same rule as the table."""
    rng = np.random.default_rng(seed)
    out = {}
    for env, _ in ORDER:
        sub = d[d.env_name == env]
        res = {"TrainSamples": [], "EncodeDim": []}
        for _ in range(B):
            pick = rng.choice(SEEDS, size=len(SEEDS), replace=True)
            bs = pd.concat([sub[sub.seed == s] for s in pick], ignore_index=True)
            for kind, groups, key, xcol, keep in (("TrainSamples", MULTS, "encode_dim_param", "train_samples", lambda g: True),
                                                  ("EncodeDim", MS, "train_samples", "z", lambda g: g >= 16000)):
                a = [p[1] for g in groups if keep(g) for p in [fit(bs[bs[key] == g][xcol], bs[bs[key] == g].test_Kloss)] if p is not None]
                if a:
                    res[kind].append(np.mean(a))
        out[env] = {k: np.percentile(v, [2.5, 97.5]) for k, v in res.items()}
        print(f"  {env:16s} alpha_m 95% [{out[env]['TrainSamples'][0]:.2f}, {out[env]['TrainSamples'][1]:.2f}]"
              f"  alpha_n 95% [{out[env]['EncodeDim'][0]:.2f}, {out[env]['EncodeDim'][1]:.2f}]", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="koopman_results_log.csv files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=0, help="number of seed-bootstrap resamples (e.g. 300)")
    args = ap.parse_args()
    d = load(args.logs)
    print(d.groupby("env_name").size().to_dict())
    import plot_style as paper_style
    rc = dict(paper_style.RC, **{"legend.fontsize": 6})
    table = {}
    for kind in ("TrainSamples", "EncodeDim"):
        groups, key, xcol = (MULTS, "encode_dim_param", "train_samples") if kind == "TrainSamples" else (MS, "train_samples", "z")
        cols = {g: plt.cm.viridis(v) for g, v in zip(groups, np.linspace(0.05, 0.95, 5))}
        with plt.rc_context(rc):
            fig, axes = plt.subplots(2, 3, figsize=(paper_style.FULL_W, 3.7), constrained_layout=True)
            for ax, (env, title) in zip(axes.flatten(), ORDER):
                sub = d[d.env_name == env]
                handles, alphas = [], {}
                for g in groups:
                    gd = sub[sub[key] == g]
                    if gd.empty:
                        continue
                    ax.plot(gd[xcol], gd.test_Kloss, "o", color=cols[g], alpha=0.35, markeredgewidth=0, ms=2.5)
                    p = fit(gd[xcol], gd.test_Kloss)
                    lab = (f"$n_{{\\mathrm{{mult}}}}={g}$" if kind == "TrainSamples" else f"$m={g // 1000}$k")
                    if p is not None:
                        x = np.geomspace(gd[xcol].min(), gd[xcol].max(), 200)
                        ax.plot(x, p[0] * x ** (-p[1]) + p[2], "-", lw=1.1, color=cols[g])
                        lab += f", $\\alpha={0.0 if abs(p[1]) < 0.05 else p[1]:.1f}$"
                        alphas[g] = p[1]
                    handles.append(Line2D([0], [0], marker="o", ms=3, ls="-", lw=1.1, color=cols[g], label=lab))
                table.setdefault(env, {})[kind] = (np.mean(list(alphas.values())) if kind == "TrainSamples"
                                                   else np.mean([a for g, a in alphas.items() if g >= 16000]))
                ax.set_title(title)
                ax.set_xscale("log", base=10 if kind == "TrainSamples" else 2)
                ax.set_yscale("log")
                ax.grid(True, which="major", ls="--")
                ax.set_xlabel("Training samples $m$" if kind == "TrainSamples" else "Latent dimension $n$")
                ax.set_ylabel("Prediction error (MSE)")
                lo, hi = ax.get_ylim()
                if kind == "EncodeDim" and env in ("G1", "Go2"):   # headroom so the legend does not hide the m=1k curve
                    hi = lo * (hi / lo) ** 1.9
                    ax.set_ylim(lo, hi)
                ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(paper_style.minor_log_label) if hi / lo < 10
                                             else matplotlib.ticker.NullFormatter())
                ax.legend(handles=handles, loc="upper right" if kind == "TrainSamples" or env in ("G1", "Go2") else "best", frameon=True)
            fig.savefig(os.path.join(args.out, f"AllEnvs_ScalingLaw_{kind}.pdf"))
            plt.close(fig)
    print("\nTable 2 (alpha_m averaged over n_mult; alpha_n averaged over m >= 16k):")
    for env, _ in ORDER:
        print(f"  {env:16s} {table[env]['TrainSamples']:.2f}  {table[env]['EncodeDim']:.2f}")
    if args.bootstrap:
        print("\nSeed-bootstrap 95% intervals:")
        bootstrap(d, B=args.bootstrap)


if __name__ == "__main__":
    main()

