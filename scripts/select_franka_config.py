"""Stage-1 selection for franka_tune.py: choose settings by the STAR-trajectory tracking error only.

Reads ../log/franka_tuning/select_results_*.csv, averages the star error over latent sizes and seeds for every
setting, picks the lowest per arm (baseline, +Aux), and writes ../log/franka_tuning/selected.json.
"""
import glob
import json
import os

import pandas as pd

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "log", "franka_tuning")
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(root, "select_results_*.csv"))], ignore_index=True)
bad = d[~d.error_star.apply(lambda v: pd.notna(v) and v != float("inf"))]
print(f"{len(d)} evaluations; {len(bad)} failed or unstable (counted as missing)")
d = d.drop(bad.index)
g = (d.groupby(["arm", "steps", "wcov", "wctrl"])
       .agg(val_star=("error_star", "mean"), runs=("error_star", "size"), fig8_not_used=("error_figure8", "mean"))
       .reset_index().sort_values(["arm", "val_star"]))
print(g.to_string(index=False))
pick = {a: g[g.arm == a].iloc[0] for a in ("base", "both")}
sel = {"baseline": {"train_steps": int(pick["base"].steps), "val_star": float(pick["base"].val_star)},
       "both": {"train_steps": int(pick["both"].steps), "cov_loss_weight": float(pick["both"].wcov),
                "ctrl_loss_weight": float(pick["both"].wctrl), "val_star": float(pick["both"].val_star)}}
json.dump(sel, open(os.path.join(root, "selected.json"), "w"), indent=2)
print(json.dumps(sel, indent=2))
