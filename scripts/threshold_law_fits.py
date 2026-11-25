"""Censored-likelihood fits of empirical sample-threshold laws for the fixed-feature study (Fig. 5c, Table in Sec. 6.1.4).

Data: log/<FFS_CSV> (default fixed_feature_study.csv, written by fixed_feature_study.py; per draw: seed, n, kappa = estimated leverage of r = [Phi(x); u], m, dG).
m50 = number of transitions at which P(||G_tilde_{n,m} - I|| <= 1/2) = 1/2 (log-interpolated); encoders that never reach
probability 1/2 within the largest m are right-censored at that m.
Model: log m50 = a + f(N, kappa) + e, e ~ N(0, s^2), N = n + n_u (n_u = 1). Censored points enter with log P(e > ...).
Laws: N, N log N, N^p, kappa log N, N log N (kappa/N)^eta.
Reports AIC and leave-one-seed-out RMS error of log m50 (natural log) on held-out uncensored encoders; the held-out
seed's thresholds are excluded from fitting, and its kappa comes from its features only (no regression draws).
The primary analysis excludes encoders whose reference Gram matrix is singular to numerical precision
(lambda_min / lambda_max < RANK_TOL = 1e-12), because they violate the full-rank assumption of Proposition 1.
For CSV files without the gram_rel_lam_min column, list such encoders as EXCLUDE="n:seed,n:seed".

Usage: python threshold_law_fits.py
"""
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "..", "log", os.environ.get("FFS_CSV", "fixed_feature_study.csv")))
d["good"] = d.dG <= 0.5
MMAX = d.m.max()
RANK_TOL = 1e-12
if "gram_rel_lam_min" in d:
    SINGULAR = {(int(n), int(s)) for (n, s), g in d.groupby(["n", "seed"]) if g.gram_rel_lam_min.iloc[0] < RANK_TOL}
else:
    SINGULAR = {tuple(int(v) for v in e.split(":")) for e in os.environ.get("EXCLUDE", "").split(",") if e}


def m50(g):
    p = g.groupby("m").good.mean()
    ms, pv = p.index.values, p.values
    if pv.max() < 0.5:
        return np.inf
    i = int(np.argmax(pv >= 0.5))
    return float(np.exp(np.interp(0.5, [pv[i - 1], pv[i]], [np.log(ms[i - 1]), np.log(ms[i])])))


r = pd.DataFrame([(n, s, g.kappa.iloc[0], m50(g)) for (n, s), g in d.groupby(["n", "seed"])],
                 columns=["n", "seed", "kappa", "m50"])
r["N"] = r.n + 1
r["cens"] = ~np.isfinite(r.m50)
r["y"] = np.log(np.where(r.cens, MMAX, r.m50))
r["singular"] = [(n, s) in SINGULAR for n, s in zip(r.n, r.seed)]

LN, LL, LK = np.log, lambda N: np.log(N * np.log(N)), None
LAWS = {  # name: (feature builder returning offset and optional regressor, has free exponent)
    "N": (lambda t: (np.log(t.N), None)),
    "N log N": (lambda t: (np.log(t.N * np.log(t.N)), None)),
    "N^p": (lambda t: (0 * t.N, np.log(t.N))),
    "kappa log N": (lambda t: (np.log(t.kappa * np.log(t.N)), None)),
    "N log N (kappa/N)^eta": (lambda t: (np.log(t.N * np.log(t.N)), np.log(t.kappa / t.N))),
}


def nll(theta, off, x, y, cens):
    a, ls = theta[0], theta[1]
    b = theta[2] if x is not None else 0.0
    mu = a + off + (b * x if x is not None else 0.0)
    s = np.exp(ls)
    z = (y - mu) / s
    ll = np.where(cens, stats.norm.logsf(z), stats.norm.logpdf(z) - ls)
    return -ll.sum()


def fit(t, law):
    off, x = LAWS[law](t)
    off, y, cens = np.asarray(off, float), t.y.values, t.cens.values
    x = None if x is None else np.asarray(x, float)
    th0 = [np.mean(y - off), np.log(0.5)] + ([0.5] if x is not None else [])
    res = optimize.minimize(nll, th0, args=(off, x, y, cens), method="Nelder-Mead",
                            options=dict(xatol=1e-8, fatol=1e-10, maxiter=20000))
    return res.x, res.fun, (off, x)


def predict(theta, t, law):
    off, x = LAWS[law](t)
    return theta[0] + np.asarray(off, float) + (theta[2] * np.asarray(x, float) if x is not None else 0.0)


def analyse(t, label):
    print(f"\n== {label}: {len(t)} encoders, {int(t.cens.sum())} censored")
    rows = []
    for law in LAWS:
        th, f, _ = fit(t, law)
        k = len(th)
        errs = []
        for s in sorted(t.seed.unique()):
            tr, te = t[t.seed != s], t[(t.seed == s) & ~t.cens]
            th_s, _, _ = fit(tr, law)
            errs += list(predict(th_s, te, law) - te.y.values)
        rows.append((law, 2 * k + 2 * f, float(np.sqrt(np.mean(np.square(errs)))), th[2] if k == 3 else np.nan))
    out = pd.DataFrame(rows, columns=["law", "AIC", "LOSO_RMS_log", "exponent"])
    print(out.round(3).to_string(index=False))
    return out


prim = r[~r.singular]
analyse(r, "all 30 encoders")
analyse(prim, "primary: without the singular encoder")
analyse(prim[~prim.cens], "uncensored only")

u = prim[~prim.cens].copy()
u["resid"] = np.log(u.m50 / (u.N * np.log(u.N)))
u["lk"] = np.log(u.kappa / u.N)
rho, p = stats.spearmanr(u.resid, u.lk)
print(f"\nrank correlation of log(m50/(N log N)) with log(kappa/N), primary uncensored: {rho:.2f} (p={p:.1g}, n={len(u)})")
rho_n, p_n = stats.spearmanr(np.log(u.N), u.lk)
print(f"rank correlation of log N with log(kappa/N): {rho_n:.2f} (p={p_n:.1g})")
# dimension-controlled: within-dimension rank correlation of log m50 with log kappa, and partial correlation
within = []
for n, g in u.groupby("n"):
    if len(g) >= 3:
        rr, pp = stats.spearmanr(np.log(g.m50), np.log(g.kappa))
        within.append((n, len(g), rr, pp))
        print(f"  n={n}: within-dimension rank corr(log m50, log kappa) = {rr:.2f} (p={pp:.2g}, {len(g)} encoders)")
u["ly"], u["lkap"], u["lN"] = np.log(u.m50), np.log(u.kappa), np.log(u.N)
dm = u.groupby("n")[["ly", "lkap"]].transform(lambda v: v - v.mean())
rr, pp = stats.pearsonr(dm.ly, dm.lkap)
print(f"pooled within-dimension correlation (dimension means removed): {rr:.2f} (p={pp:.1g}, n={len(u)})")
