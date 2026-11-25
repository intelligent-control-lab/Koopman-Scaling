"""Figure for the fixed-feature test of Proposition 1 / Remark 1 (log/fixed_feature_study.csv).

(a) sampling error vs m, with an m^{-1/2} reference;
(b) probability of the event ||dG|| <= 1/2 vs m / (N ln N), N = n + 1 regressors;
(c) one-step state error vs m: the best latent dimension grows with m;
(d) transitions m_50 at which the event holds with probability 1/2, vs N = n + 1, with the censored fits of the
    N log N and N laws (threshold_law_fits.py; the rank-deficient encoder is shown but excluded).

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


sys.path.insert(0, "/Users/abu/Documents/Code/Others/koopman/koopman_scaling_paper/tools")
import paper_style  # noqa: E402
rc = dict(paper_style.RC, **{"legend.fontsize": 6})
with plt.rc_context(rc):
    fig, axs = plt.subplots(1, 3, figsize=(paper_style.FULL_W, 2.05), constrained_layout=True)
    ax = [axs[0], None, axs[1], axs[2]]
    for n, g in d.groupby("n"):
        per_seed = g.groupby(["m", "seed"]).agg(es=("eps_samp", "median"), mt=("mse_tot", "median"),
                                                 pg=("good", "mean")).reset_index()
        s = per_seed.groupby("m").agg(es=("es", "median"), es_lo=("es", "min"), es_hi=("es", "max"),
                                      mt=("mt", "median"), pg=("pg", "mean"))
        m = s.index.values
        lab = f"$n={n}$"
        ax[0].plot(m, s.es, "o-", color=cols[n], ms=2, lw=1, label=lab)
        ax[0].fill_between(m, s.es_lo, s.es_hi, color=cols[n], alpha=0.15, lw=0)
        ax[2].plot(m, s.mt, "o-", color=cols[n], ms=2, lw=1, label=lab)
        ax[2].axhline(g.mse_proj.median(), color=cols[n], ls=":", lw=1)
    mref = np.array([80, 8000])
    ax[0].plot(mref, 0.03 * (mref / 80) ** -0.5, "k--", lw=1, label="$\\propto m^{-1/2}$")
    import contextlib, io, runpy
    with contextlib.redirect_stdout(io.StringIO()):     # same thresholds and censored fits as threshold_law_fits.py
        T = runpy.run_path(os.path.join(HERE, "threshold_law_fits.py"))
    r, prim = T["r"], T["prim"]
    ok = ~prim.cens
    sc = ax[3].scatter(prim.N[ok], prim.m50[ok], c=np.log10(prim.kappa[ok] / prim.N[ok]), cmap="plasma", s=7,
                       vmin=0.6, vmax=1.9, label="per encoder", zorder=3)
    ax[3].scatter(prim.N[~ok], np.full((~ok).sum(), d.m.max()), marker="^", facecolors="none",
                  edgecolors="k", s=12, label="not reached", zorder=3)
    sing = r[r.singular]
    ax[3].scatter(sing.N, np.where(sing.cens, d.m.max(), sing.m50), marker="x", color="0.5", s=12,
                  label="rank-deficient", zorder=3)
    cb = fig.colorbar(sc, ax=ax[3], pad=0.01); cb.set_label("$\\log_{10}(\\hat\\kappa_r/N)$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    x = np.geomspace(prim.N.min(), prim.N.max(), 100)
    th1 = T["fit"](prim, "N log N")[0]
    th2 = T["fit"](prim, "N")[0]
    c1, c2 = np.exp(th1[0]), np.exp(th2[0])
    ax[3].plot(x, c1 * x * np.log(x), "-", color="C3", label=f"${c1:.1f}\\,N\\log N$")
    ax[3].plot(x, c2 * x, ":", color="C0", label=f"${c2:.0f}\\,N$")
    ax[0].set(xscale="log", yscale="log", xlabel="i.i.d. transitions $m$", ylabel="Sampling error (relative)",
              title="(a) Sampling error")
    ax[2].set(xscale="log", yscale="log", xlabel="i.i.d. transitions $m$", ylabel="One-step state MSE (relative)",
              title="(b) Total error; dotted: projection", ylim=(1.2e-4, 4e-4))
    ax[3].set(xscale="log", yscale="log", xlabel="Regressor dimension $N=n+1$", ylabel="$m_{50}$ (transitions)",
              title="(c) Data threshold")
    import matplotlib.ticker as mt
    Ns = [v + 1 for v in ns]
    ax[3].xaxis.set_major_locator(mt.FixedLocator(Ns)); ax[3].xaxis.set_major_formatter(mt.FixedFormatter([str(v) for v in Ns]))
    ax[3].xaxis.set_minor_formatter(mt.NullFormatter())
    for a in (ax[0], ax[2]):
        a.yaxis.set_minor_formatter(mt.FuncFormatter(paper_style.minor_log_label) if a.get_ylim()[1] / a.get_ylim()[0] < 10
                                    else mt.NullFormatter())
    for a in axs:
        a.grid(True, which="major", ls="--")
        a.legend(frameon=True, ncol=1)
    fig.savefig(sys.argv[1])
