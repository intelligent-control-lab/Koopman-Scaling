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
SUFFIX = {"base": "Baseline", "cov": "+Cov", "ctrl": "+Ctrl", "both": "+Aux"}   # aux_final_<ENV>_<arm>
d = tests.merge(models[["model_path", "project", "train_samples", "encode_dim_param", "encode_dim", "seed"]],
                on="model_path")
d["arm"] = d.project.str.rsplit("_", n=1).str[-1].map(SUFFIX)
print(f"{len(d)} evaluated models;", d.groupby("arm").size().to_dict())
ARMS = ["Baseline", "+Cov", "+Ctrl", "+Aux"]

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
diff = seedmean["+Aux"] - seedmean["Baseline"]
print(f"  +Aux - Baseline per seed: {diff.round(2).tolist()}  (positive in {(diff > 0).sum()}/{len(diff)} seeds)")

try:
    import plot_style as paper_style
    env = d.project.iloc[0].split("_")[2].lower()
    n_x = {"g1": 53, "go2": 35}[env]   # latent dimension n = n_x + n_enc
    paper_style.control_panels(left, right, n_x, "Survival steps", os.path.join(out, f"{env}_control_tuned.pdf"))
    print("\nwrote", os.path.join(out, f"{env}_control_tuned.pdf"))
except ImportError:
    pass
