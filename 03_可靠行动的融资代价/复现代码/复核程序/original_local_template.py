import numpy as np
from scipy.optimize import brentq

def old_build(eta):
    z = np.arange(1.0, 5.0)
    m = np.full(4, 0.25)
    x = np.arange(1.0, 4.0)
    u = np.array([5.0, -8.0, 3.0]) / 8
    V = np.array([np.ones(4), -np.expm1(-z), z, z * z])
    b = 0.015
    worlds = []
    for sign in [1, -1]:
        sig = (1 + sign * 0.9 * u) / 3
        fastz, fastw = (x / eta, b * eta ** 2 * sig)
        rhs = np.array([fastw.sum(), fastw @ -np.expm1(-fastz), 0.0, fastw @ (fastz * fastz)])
        sloww = m - np.linalg.solve(V, rhs)
        assert min(sloww) > 0 and min(fastw) > 0
        worlds.append((np.r_[z, fastz], np.r_[sloww, fastw]))
    c = float(m @ z)
    public = np.array([1.0, m @ -np.expm1(-z), c, m @ (z * z)])
    residual = 0.0
    opts = []
    for zz, ww in worlds:
        actual = np.array([ww.sum(), ww @ -np.expm1(-zz), ww[:4] @ z, ww @ (zz * zz)])
        residual = max(residual, float(max(abs(actual - public))))
        assert residual < 2e-12
        opt = brentq(lambda q: ww @ (zz * np.exp(-q * zz)) - c, 0, 1, xtol=1e-15)
        opts.append(opt)

    def profit(s, q):
        zz, ww = worlds[s]
        return float(ww @ -np.expm1(-q * zz) - c * q)
    gaproot = brentq(lambda q: profit(0, opts[0]) - profit(0, q) - profit(1, opts[1]) + profit(1, q), min(opts), max(opts), xtol=1e-15)
    regret = profit(0, opts[0]) - profit(0, gaproot)
    assert regret > 0
    return (worlds, c, public, residual, opts, regret)
