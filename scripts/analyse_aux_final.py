"""Analyse the G1 auxiliary-loss tuning results (final stage, test trajectories 0-999).

Usage:  python analyse_aux_final.py <dir with final_models_*.csv and final_test_*.csv> [out_dir]
Prints the per-sweep curves, the relative changes in the format of the paper's Table 4, the m=140k values
with seed spread, and writes <env>_control_tuned.pdf in the style of the paper's Fig. 6.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else src
models = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(src, "final_models_*.csv")))
                    if "chunk" not in f], ignore_index=True)
tests = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(src, "final_test_*.csv")))
                   if "chunk" not in f], ignore_index=True)
if "failed" in tests:
    print("failed evaluations:", int(tests.failed.fillna(0).sum()))
    tests = tests[tests.failed.fillna(0) != 1]
SUFFIX = {"base": "Baseline", "cov": "+Cov", "ctrl": "+Ctrl", "both": "+Both"}   # aux_final_<ENV>_<arm>
d = tests.merge(models[["model_path", "project", "train_samples", "encode_dim_param", "encode_dim", "seed"]],
                on="model_path")
d["arm"] = d.project.str.rsplit("_", n=1).str[-1].map(SUFFIX)
print(f"{len(d)} evaluated models;", d.groupby("arm").size().to_dict())
ARMS = ["Baseline", "+Cov", "+Ctrl", "+Both"]

cell = d.groupby(["arm", "train_samples", "encode_dim"]).mean_survival_steps.mean().reset_index()
left = cell.groupby(["arm", "train_samples"]).mean_survival_steps.mean().unstack(0)[ARMS]
right = cell.groupby(["arm", "encode_dim"]).mean_survival_steps.mean().unstack(0)[ARMS]
print("\nSurvival vs training samples (mean over latent sizes of seed means):\n", left.round(2).to_string())
print("\nSurvival vs latent dimension (mean over training-set sizes of seed means):\n", right.round(2).to_string())

print("\nRelative change vs baseline (mean over the 5 points of each sweep), as in Table 4:")
for name, panel in (("data m", left), ("latent n", right)):
    rel = {a: 100 * ((panel[a] - panel["Baseline"]) / panel["Baseline"]).mean() for a in ARMS[1:]}
    print(f"  {name:9s}", "  ".join(f"{a} {v:+.1f}%" for a, v in rel.items()))

big = d[d.train_samples == d.train_samples.max()]
seedmean = big.groupby(["arm", "seed"]).mean_survival_steps.mean().unstack(0)[ARMS]
print(f"\nm={int(d.train_samples.max())}: mean over latent sizes, per seed:\n", seedmean.round(2).to_string())
print("  mean +- std over seeds:", {a: f"{seedmean[a].mean():.1f} +- {seedmean[a].std(ddof=1):.1f}" for a in ARMS})
diff = seedmean["+Both"] - seedmean["Baseline"]
print(f"  +Both - Baseline per seed: {diff.round(2).tolist()}  (positive in {(diff > 0).sum()}/{len(diff)} seeds)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"Baseline": ("tab:blue", "-"), "+Cov": ("tab:orange", "--"), "+Ctrl": ("tab:green", "--"),
             "+Both": ("tab:red", "-")}
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
    n_x = {"g1": 53, "go2": 35}[d.project.iloc[0].split("_")[2].lower()]   # latent dimension n = n_x + n_enc
    for a in ARMS:
        c, ls = style[a]
        ax[0].plot(left.index, left[a], ls, color=c, marker="o", label=a, lw=2)
        ax[1].plot(right.index + n_x, right[a], ls, color=c, marker="o", label=a, lw=2)
    for k, x in ((0, "Training windows $m_{\\mathrm{win}}$"), (1, "Latent dimension $n$")):
        ax[k].set_xscale("log", base=10 if k == 0 else 2)
        ax[k].set_xlabel(x, fontsize=13)
        ax[k].set_ylabel("Survival Steps", fontsize=13)
        ax[k].grid(True, ls="--", alpha=0.4)
        ax[k].legend(ncol=2, fontsize=9)
    fig.tight_layout()
    env = d.project.iloc[0].split("_")[2].lower()
    fig.savefig(os.path.join(out, f"{env}_control_tuned.pdf"), bbox_inches="tight")
    print("\nwrote", os.path.join(out, f"{env}_control_tuned.pdf"))
except ImportError:
    pass
