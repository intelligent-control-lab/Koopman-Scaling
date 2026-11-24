"""Independent reference pool for the fixed-feature study: new damped-pendulum trajectories from the same
distribution as the training dataset (initial angle in +-90 deg, velocity in +-3 rad/s, i.i.d. uniform torques).
Usage: python make_reference_pool.py OUT.npz [N_TRAJ] [SEED]"""
import random
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utility"))
import dataset as D  # noqa: E402

out, n, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16000, int(sys.argv[3]) if len(sys.argv) > 3 else 20261124
random.seed(seed); np.random.seed(seed)
c = D.DampingPendulumDataCollector()
c.random_state = lambda: np.array([random.uniform(-np.pi / 2, np.pi / 2), random.uniform(-3.0, 3.0)])
W = c.collect_koopman_data(n, 15)          # (16, n, 3), rows [u_t, theta_t, omega_t]
np.savez(out, windows=W)
print("saved", out, W.shape)
