"""Figure for the fixed-feature test of Proposition 1 (log/fixed_feature_study.csv).

(a) sampling error vs m, with an m^{-1/2} reference;
(b) probability of the good event ||dG|| <= 1/2 vs m / (n ln n);
(c) one-step state error vs m: the best latent dimension grows with m.

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
d = pd.read_csv(os.path.join(HERE, "..", "log", "fixed_feature_study.csv"))
d["good"] = d.dG <= 0.5
cols = {4: plt.cm.viridis(0.1), 10: plt.cm.viridis(0.5), 34: plt.cm.viridis(0.85)}
rc = {"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
      "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7, "pdf.fonttype": 42}
with plt.rc_context(rc):
    fig, ax = plt.subplots(1, 3, figsize=(10, 2.9), constrained_layout=True)
    for n, g in d.groupby("n"):
        per_seed = g.groupby(["m", "seed"]).agg(es=("eps_samp", "median"), et=("eps_tot", "median"),
                                                 pg=("good", "mean")).reset_index()
        s = per_seed.groupby("m").agg(es=("es", "median"), es_lo=("es", "min"), es_hi=("es", "max"),
                                      et=("et", "median"), et_lo=("et", "min"), et_hi=("et", "max"),
                                      pg=("pg", "mean"))
        m = s.index.values
        lab = f"$n={n}$"
        ax[0].plot(m, s.es, "o-", color=cols[n], ms=3, label=lab)
        ax[0].fill_between(m, s.es_lo, s.es_hi, color=cols[n], alpha=0.2, lw=0)
        ax[1].plot(m / ((n + 1) * np.log(n + 1)), s.pg, "o-", color=cols[n], ms=3, label=lab)
        ax[2].plot(m, s.et, "o-", color=cols[n], ms=3, label=lab)
        ax[2].fill_between(m, s.et_lo, s.et_hi, color=cols[n], alpha=0.2, lw=0)
    mref = np.array([80, 8000])
    ax[0].plot(mref, 0.03 * (mref / 80) ** -0.5, "k--", lw=1, label="$\\propto m^{-1/2}$")
    ax[0].set(xscale="log", yscale="log", xlabel="i.i.d. transitions $m$", ylabel="sampling error (relative)",
              title="(a) Sampling error")
    ax[1].set(xscale="log", xlabel="$m/(n\\ln n)$", ylabel="P($\\|\\Delta\\tilde G\\|\\leq 1/2$)",
              title="(b) Stability event of Prop. 1")
    ax[2].set(xscale="log", yscale="log", xlabel="i.i.d. transitions $m$", ylabel="one-step state error (relative)",
              title="(c) Total error", ylim=(0.008, 0.017))
    for a in ax:
        a.grid(True, which="both", ls="--", alpha=0.4)
        a.legend(frameon=True)
    fig.savefig(sys.argv[1])
