"""Convert portable dataset caches (<name>.cache.npz) into the training caches (<name>.pt) that train_model.py reads.

A .pt file pickled by one NumPy version may not load with another; the .npz caches are version-independent.
Each .npz holds Ktrain_data, Kval_data, Ktest_data and optional normalization statistics stored as ns_<key>.

Usage (from scripts/):  python convert_dataset_cache.py [DATA_DIR]     (default: ../data/datasets)
"""
import glob
import os
import sys

import numpy as np
import torch

data_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "datasets")
for src in sorted(glob.glob(os.path.join(data_dir, "*.cache.npz"))):
    dst = src[: -len(".cache.npz")] + ".pt"
    if os.path.exists(dst):
        continue
    z = np.load(src)
    norm = {k[3:]: z[k] for k in z.files if k.startswith("ns_")}
    torch.save({"Ktrain_data": z["Ktrain_data"], "Kval_data": z["Kval_data"], "Ktest_data": z["Ktest_data"],
                "norm_stats": norm or None}, dst)
    print("wrote", dst)
