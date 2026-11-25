"""Export the per-encoder data of Fig. 5c for the project page (docs/assets/fig5c.json).

For every frozen pendulum encoder of the fixed-feature study: latent size n, regressor size N = n + 1, seed,
estimated leverage kappa_r, observed threshold m50 (null if not reached), status, and the thresholds predicted for it by
the N log N law and by the leverage-adjusted law N log N (kappa_r/N)^eta, each fitted without the encoder's own seed
(leave-one-seed-out, as in threshold_law_fits.py).

Usage (from scripts/):  python export_fig5c_web.py
"""
import contextlib
import io
import json
import os

import numpy as np

with contextlib.redirect_stdout(io.StringIO()):
    import threshold_law_fits as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "assets", "fig5c.json")
prim = T.r[~T.r.singular]
laws = {"nlogn": "N log N", "leverage": "N log N (kappa/N)^eta"}
pred = {k: {} for k in laws}
for s in sorted(prim.seed.unique()):
    train, test = prim[prim.seed != s], T.r[T.r.seed == s]
    for key, law in laws.items():
        th, _, _ = T.fit(train, law)
        for (n, seed), v in zip(zip(test.n, test.seed), T.predict(th, test, law)):
            pred[key][(int(n), int(seed))] = float(np.exp(v))
full = {key: T.fit(prim, law)[0].tolist() for key, law in laws.items()}
rows = []
for _, e in T.r.sort_values(["n", "seed"]).iterrows():
    k = (int(e.n), int(e.seed))
    status = "rank-deficient" if e.singular else ("not reached" if e.cens else "reached")
    rows.append(dict(n=k[0], N=int(e.N), seed=k[1], kappa=None if e.singular else round(float(e.kappa), 1),
                     m50=None if e.cens else round(float(e.m50), 1), status=status,
                     pred_nlogn=None if e.singular else round(pred["nlogn"][k], 1),
                     pred_leverage=None if e.singular else round(pred["leverage"][k], 1)))
json.dump(dict(m_max=float(T.MMAX), fit_full=full, encoders=rows), open(OUT, "w"), indent=1)
print(f"wrote {OUT}: {len(rows)} encoders")
