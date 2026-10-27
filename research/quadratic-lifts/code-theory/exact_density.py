"""Exact rational discrete-Darboux search (fixed rational h) returning the densities P themselves."""
import sympy as sp, random
from itertools import combinations_with_replacement
def densities(F, vars, deg, h, seed=3, extra=25, rng_den=7):
    z = sp.Matrix(vars); Fm = sp.Matrix(F); J = Fm.jacobian(vars); n = len(vars)
    mons = [sp.Integer(1)]
    for k in range(1, deg + 1):
        for idx in combinations_with_replacement(vars, k): mons.append(sp.Mul(*idx))
    random.seed(seed); rows = []
    while len(rows) < len(mons) + extra:
        zv = [sp.Rational(random.randint(-9, 9), random.randint(1, rng_den)) for _ in range(n)]
        sub = dict(zip(vars, zv)); M = sp.eye(n) - h/2*J.subs(sub)
        if M.det() == 0: continue
        zn = sp.Matrix(zv) + h*M.LUsolve(Fm.subs(sub)); Jn = J.subs(dict(zip(vars, list(zn))))
        det = (sp.eye(n) + h/2*Jn).det()/M.det(); s1 = dict(zip(vars, list(zn)))
        rows.append([m.subs(s1) - det*m.subs(sub) for m in mons])
    ns = sp.Matrix(rows).nullspace()
    return [sp.factor(sum(c*m for c, m in zip(v, mons))) for v in ns]
