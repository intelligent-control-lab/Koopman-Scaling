"""Analyse the Franka MPC-setting study (franka_mpc_grid.py).

Selection uses the STAR trajectory only: for each model type (arm), pick the MPC setting with the lowest mean star
error over all of that arm's models (errors capped at CAP, so failed runs count as large errors). The figure-8
error under the selected setting is the test metric. A shared setting (best mean star error over all Koopman arms
together) is reported as a check.

Usage: python analyse_franka_mpc_grid.py GRID_DIR [OUT_PREFIX]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

CAP = 1.0
d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(sys.argv[1], "grid_*.csv")))], ignore_index=True)
d = d.drop_duplicates(["model_path", "cfg"])
for c in ("err_star", "err_figure8"):
    d[c + "_c"] = d[c].clip(upper=CAP).fillna(CAP)
key = ["H", "R", "vel", "Qv"]
orig = (d.H == 15) & (d.R == 0) & (d.vel == "zero") & (d.Qv == 1.0)
print(f"{d.model_path.nunique()} models, {d.cfg.nunique()} settings")

sel = {}
for arm, g in d.groupby("arm"):
    s = g.groupby("cfg").err_star_c.mean()
    sel[arm] = int(s.idxmin())
    c = g[g.cfg == sel[arm]].iloc[0]
    print(f"{arm:5s} selected cfg {sel[arm]:3d} H={c.H} R={c.R} vel={c.vel} Qv={c.Qv}: star {s.min():.3f} "
          f"(original setting {g[orig.loc[g.index]].err_star_c.mean():.3f})")
koop = d[d.arm != "edmd"]
shared = int(koop.groupby("cfg").err_star_c.mean().idxmin())
c = d[d.cfg == shared].iloc[0]
print(f"shared cfg {shared}: H={c.H} R={c.R} vel={c.vel} Qv={c.Qv}")


def table(choice, label):
    rows = []
    for arm, g in d.groupby("arm"):
        cfg = choice[arm] if isinstance(choice, dict) else choice
        g = g[g.cfg == cfg]
        for m, h in g.groupby("m"):
            rows.append(dict(sel=label, arm=arm, m=m, f8=h.err_figure8_c.mean(), f8_sd=h.err_figure8_c.std(),
                             star=h.err_star_c.mean(), n=len(h), fails=int((h.err_figure8 >= CAP).sum())))
    return pd.DataFrame(rows)


res = pd.concat([table(sel, "per-arm"), table(shared, "shared"),
                 table({a: int(d[orig & (d.arm == a)].cfg.iloc[0]) for a in d.arm.unique()}, "original")])
pd.set_option("display.width", 200)
print(res.pivot_table(index=["sel", "m"], columns="arm", values="f8").round(3))
for label in ("per-arm", "shared"):
    cfgs = sel if label == "per-arm" else {a: shared for a in d.arm.unique()}
    t = {a: d[(d.arm == a) & (d.cfg == cfgs[a]) & (d.m == 140000)] for a in d.arm.unique()}
    b = t["base"].set_index(["nm", "seed"]).err_figure8_c
    for a in ("cov", "ctrl", "both"):
        x = t[a].set_index(["nm", "seed"]).err_figure8_c
        j = b.index.intersection(x.index)
        print(f"{label} m=140k {a} vs base: {x.mean():.3f} vs {b.mean():.3f}, paired p={stats.ttest_rel(x[j], b[j]).pvalue:.3g}")
    e = t["edmd"]
    print(f"{label} EDMD nm=16: {e[e.nm == 16].err_figure8_c.mean():.3f} +- {e[e.nm == 16].err_figure8_c.std():.3f}")
    kb = d[(d.arm == "base") & (d.cfg == cfgs["base"])]
    print(f"{label} base: Spearman(test_Kloss, fig8) = {stats.spearmanr(kb.test_Kloss, kb.err_figure8_c)[0]:.2f}, "
          f"Spearman(star, fig8) = {stats.spearmanr(kb.err_star_c, kb.err_figure8_c)[0]:.2f}")
if len(sys.argv) > 2:
    res.to_csv(sys.argv[2] + "_summary.csv", index=False)
    d[d.cfg.isin(set(sel.values()) | {shared})].to_csv(sys.argv[2] + "_selected_rows.csv", index=False)
