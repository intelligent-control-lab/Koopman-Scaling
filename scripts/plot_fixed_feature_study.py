"""Figure for the fixed-feature test of Proposition 1 / Remark 1 (log/fixed_feature_study.csv).

(a) sampling error vs m, with an m^{-1/2} reference;
(b) probability of the event ||dG|| <= 1/2 vs m / (N ln N), N = n + 1 regressors;
(c) one-step state error vs m: the best latent dimension grows with m;
(d) transitions m_50 at which the event holds with probability 1/2, vs N, with the fitted N ln N and N laws.

Usage: python plot_fixed_feature_study.py OUT.pdf
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "log", os.environ.get("FFS_CSV", "fixed_feature_study_v2.csv")))
d["good"] = d.dG <= 0.5
ns = sorted(d.n.unique())
cols = {n: plt.cm.viridis(v) for n, v in zip(ns, np.linspace(0.05, 0.9, len(ns)))}


def m50(g):
    p = g.groupby("m").good.mean()
    ms, pv = p.index.values, p.values
    if pv.max() < 0.5:
        return np.inf        # censored: the event is not reached within the largest m
    i = int(np.argmax(pv >= 0.5))
    return float(np.exp(np.interp(0.5, [pv[i - 1], pv[i]], [np.log(ms[i - 1]), np.log(ms[i])])))


rc = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
      "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 6.5, "pdf.fonttype": 42}
with plt.rc_context(rc):
    fig, ax = plt.subplots(1, 4, figsize=(13, 2.9), constrained_layout=True)
    for n, g in d.groupby("n"):
        per_seed = g.groupby(["m", "seed"]).agg(es=("eps_samp", "median"), mt=("mse_tot", "median"),
                                                 pg=("good", "mean")).reset_index()
        s = per_seed.groupby("m").agg(es=("es", "median"), es_lo=("es", "min"), es_hi=("es", "max"),
                                      mt=("mt", "median"), pg=("pg", "mean"))
        m = s.index.values
        lab = f"$n={n}$"
        ax[0].plot(m, s.es, "o-", color=cols[n], ms=2.5, label=lab)
        ax[0].fill_between(m, s.es_lo, s.es_hi, color=cols[n], alpha=0.15, lw=0)
        ax[1].plot(m / ((n + 1) * np.log(n + 1)), s.pg, "o-", color=cols[n], ms=2.5, label=lab)
        ax[2].plot(m, s.mt, "o-", color=cols[n], ms=2.5, label=lab)
        ax[2].axhline(g.mse_proj.median(), color=cols[n], ls=":", lw=1)
    mref = np.array([80, 8000])
    ax[0].plot(mref, 0.03 * (mref / 80) ** -0.5, "k--", lw=1, label="$\\propto m^{-1/2}$")
    r = pd.DataFrame([(n, s, g.kappa.iloc[0], m50(g)) for (n, s), g in d.groupby(["n", "seed"])],
                     columns=["n", "seed", "kappa", "m50"])
    N = r.n + 1
    fin = np.isfinite(r.m50)
    sc = ax[3].scatter(N[fin], r.m50[fin], c=np.log10(r.kappa[fin] / N[fin]), cmap="plasma", s=14, vmin=0.7, vmax=1.9,
                       label="per seed", zorder=3)
    ax[3].scatter(N[~fin], np.full((~fin).sum(), d.m.max()), marker="^", facecolors="none",
                  edgecolors="k", s=22, label="not reached", zorder=3)
    cb = fig.colorbar(sc, ax=ax[3], pad=0.01); cb.set_label("$\\log_{10}(\\hat\\kappa_r/N)$", fontsize=7)
    med = r.groupby("n").m50.median()
    Nm = med.index.values + 1
    x = np.geomspace(Nm.min(), Nm.max(), 100)
    ok = fin
    c1 = np.exp(np.mean(np.log(r.m50[ok]) - np.log(N[ok] * np.log(N[ok]))))
    c2 = np.exp(np.mean(np.log(r.m50[ok]) - np.log(N[ok])))
    ax[3].plot(x, c1 * x * np.log(x), "-", color="C3", label=f"${c1:.1f}\\,N\\log N$")
    ax[3].plot(x, c2 * x, ":", color="C0", label=f"${c2:.0f}\\,N$")
    ax[0].set(xscale="log", yscale="log", xlabel="i.i.d. transitions $m$", ylabel="sampling error (relative)",
              title="(a) Sampling error")
    ax[1].set(xscale="log", xlabel="$m/(N\\log N)$, $N=n+1$", ylabel="P($\\|\\Delta\\tilde G\\|\\leq 1/2$)",
              title="(b) Stability event of Prop. 1")
    ax[2].set(xscale="log", yscale="log", xlabel="i.i.d. transitions $m$", ylabel="one-step state MSE (relative)",
              title="(c) Total error; dotted: projection", ylim=(1.2e-4, 4e-4))
    ax[3].set(xscale="log", yscale="log", xlabel="regressor dimension $N$", ylabel="$m_{50}$ (transitions)",
              title="(d) Data threshold")
    for a in ax:
        a.grid(True, which="both", ls="--", alpha=0.4)
        a.legend(frameon=True, ncol=1)
    fig.savefig(sys.argv[1])
