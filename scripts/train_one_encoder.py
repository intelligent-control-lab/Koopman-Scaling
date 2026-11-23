"""Train one damped-pendulum encoder exactly as split_study.py does, for the fixed-feature study.

Usage (from scripts/):  python train_one_encoder.py SEED MULT [M]
Trains on M windows (default 4000) in the per-seed order of split_study.permutation and writes
../log/<page_grid_m{M}_k{MULT} | split_s{SEED}_m{M}_k{MULT}>/best_models/*.pth, the names fixed_feature_study.py reads.
"""
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "utility"))
import dataset as D  # noqa: E402
import export_page_demo as E  # noqa: E402
from split_study import SEEDS, patch_collector, permutation  # noqa: E402


def main():
    seed, mult = int(sys.argv[1]), int(sys.argv[2])
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    import torch
    n_windows = np.asarray(torch.load(E.DATASET, weights_only=False)["Ktrain_data"]).shape[1]
    perm = permutation(seed, n_windows)
    project = f"page_grid_m{m}_k{mult}" if seed == SEEDS[0] else f"split_s{seed}_m{m}_k{mult}"
    original = patch_collector(perm)
    try:
        import train_model as tm
        D.DampingPendulumDataCollector.random_state = lambda self: np.array(
            [random.uniform(-np.pi / 2, np.pi / 2), random.uniform(-3.0, 3.0)])
        tm.train(project_name=project, env_name="DampingPendulum", max_train_samples=16000,
                 train_samples=m, val_samples=2000, test_samples=2000, Ksteps=15,
                 train_steps=15000, encode_dim=mult, hidden_layers=2, hidden_dim=64,
                 gamma=0.8, seed=seed, batch_size=64, val_step=1000, lr_step=1000,
                 lr_gamma=0.9, max_norm=0.1, use_control_loss=True,
                 use_covariance_loss=True, multiply_encode_by_input_dim=True)
    finally:
        D.KoopmanDatasetCollector.get_data = original
    print(f"[encoder] done seed={seed} mult={mult} -> {project}", flush=True)


if __name__ == "__main__":
    main()
