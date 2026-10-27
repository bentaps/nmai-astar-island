"""Algebraic entropy test: degrees of iterates of a Kahan map restricted to a random line z(t) = a + b t, computed
exactly mod a prime. For g(t) = l . Phi^k(a + b t) (random linear functional l), the degree d_k of the rational
function g is found from the nullity of the Pade system N(t_i) - g(t_i) D(t_i) = 0 with deg N, deg D <= m:
nullity = m + 1 - d_k when m >= d_k. Polynomial growth of d_k = integrable; exponential growth = not integrable."""
import random, numpy as np
from darboux_modp import ModField, kahan_mod, nullity_mod, PR

def iterate_values(fld, h, a, b, l, K, ts):
    vals = np.zeros((K, len(ts)), dtype=np.int64); good = np.ones(len(ts), bool)
    for i, t in enumerate(ts):
        z = [(ai + bi*t) % PR for ai, bi in zip(a, b)]
        for k in range(K):
            out = kahan_mod(fld, z, h)
            if out is None: good[i] = False; break
            z = out[0]; vals[k, i] = sum(li*zi for li, zi in zip(l, z)) % PR
    return vals, good

def degree_of(values, ts, m):
    rows = []
    for g, t in zip(values, ts):
        pw = [pow(t, j, PR) for j in range(m + 1)]
        rows.append(pw + [(-int(g)*x) % PR for x in pw])
    nul = nullity_mod(rows)
    return (m + 1 - nul) if nul > 0 else None

def degree_sequence(F, vars, K, mmax, seed=0):
    fld = ModField(F, vars); rng = random.Random(seed); n = len(vars)
    h = rng.randrange(1, PR); a = [rng.randrange(PR) for _ in range(n)]; b = [rng.randrange(PR) for _ in range(n)]
    l = [rng.randrange(PR) for _ in range(n)]
    ts = [rng.randrange(PR) for _ in range(2*mmax + 30)]
    vals, good = iterate_values(fld, h, a, b, l, K, ts)
    ts = [t for t, g in zip(ts, good) if g]; vals = vals[:, good]
    degs = []
    for k in range(K):
        m = 8
        while True:
            d = degree_of(vals[k, :2*m + 12], ts[:2*m + 12], m)
            if d is not None or m >= mmax: break
            m = min(2*m, mmax)
        degs.append(d if d is not None else f">{mmax}")
        if d is None: break
    return degs
