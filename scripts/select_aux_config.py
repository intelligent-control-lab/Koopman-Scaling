"""Stage-1 selection for run_aux_tuning.sh: choose training settings by VALIDATION survival only.

Reads ../log/aux_tuning_<env>/select_models.csv (training logs) and select_val_r*.csv (validation MPC
results on reference trajectories 1000-1499). For each arm (baseline, +Aux) it averages survival steps
over latent sizes and seeds and picks the best setting. The MPC input weight r is a controller choice
shared by all arms: it is picked by the mean of the two arms' best validation survival.
Writes ../log/aux_tuning_<env>/selected.json and prints the full validation table.

Usage (from scripts/):  python select_aux_config.py --env G1
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="G1")
    root = os.path.join("..", "log", f"aux_tuning_{ap.parse_args().env}")
    models = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(root, "select_models*.csv")))],
                       ignore_index=True)
    models["arm"] = models.project.str.contains("_both_").map({True: "both", False: "baseline"})
    if "ctrl_loss_weight" not in models:
        raise SystemExit("training logs lack ctrl_loss_weight; use the current train_model.py")
    keys = ["arm", "train_steps", "cov_loss_weight", "ctrl_loss_weight"]
    tables = []
    val = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(root, "select_val_r*.csv")))],
                    ignore_index=True)
    if "failed" in val:
        bad = val[val.failed.fillna(0) == 1]
        print(f"[select] {len(bad)} evaluations failed (hang or error) and are excluded:", *bad.model_path, sep="\n  ")
        val = val[val.failed.fillna(0) != 1]
    for _, part in val.groupby("mpc_r"):
        res = part.merge(models[["model_path"] + keys + ["encode_dim_param", "seed"]], on="model_path")
        res.loc[res.arm == "baseline", ["cov_loss_weight", "ctrl_loss_weight"]] = float("nan")
        g = (res.groupby(keys + ["mpc_r"], dropna=False)
                .agg(val_survival=("mean_survival_steps", "mean"), runs=("mean_survival_steps", "size"))
                .reset_index())
        tables.append(g)
    table = pd.concat(tables, ignore_index=True).sort_values(["arm", "val_survival"], ascending=[True, False])
    print(table.to_string(index=False))

    best_r = (table.groupby(["mpc_r", "arm"]).val_survival.max().groupby("mpc_r").mean().idxmax())
    pick = {arm: table[(table.arm == arm) & (table.mpc_r == best_r)].iloc[0] for arm in ("baseline", "both")}
    sel = {"mpc_r": float(best_r),
           "baseline": {"train_steps": int(pick["baseline"].train_steps),
                        "val_survival": float(pick["baseline"].val_survival)},
           "both": {"train_steps": int(pick["both"].train_steps),
                    "cov_loss_weight": float(pick["both"].cov_loss_weight),
                    "ctrl_loss_weight": float(pick["both"].ctrl_loss_weight),
                    "val_survival": float(pick["both"].val_survival)}}
    with open(os.path.join(root, "selected.json"), "w") as fh:
        json.dump(sel, fh, indent=2)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
