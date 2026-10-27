"""Exact (rational) discrete-Darboux search for an invariant-measure density 1/P of the Kahan map,
P(z') = det(DPhi(z)) P(z), for the non-symmetric quartic lift. Linear system over Q."""
import sympy as sp, random
from itertools import combinations_with_replacement
q, p, y = sp.symbols('q p y'); z = sp.Matrix([q, p, y]); h = sp.Rational(1, 5)
F = sp.Matrix([p, -(y/3 + 2*q*(y/2 + q/3 + sp.Rational(1, 2))), 2*q*p]); J = F.jacobian(z)
def kahan_and_det(zv):
    sub = dict(zip((q, p, y), zv))
    M = sp.eye(3) - h/2*J.subs(sub)
    zn = sp.Matrix(zv) + h*M.LUsolve(F.subs(sub))
    Jn = J.subs(dict(zip((q, p, y), list(zn))))
    det = (M.inv()*(sp.eye(3) + h/2*Jn)).det()
    return list(zn), sp.nsimplify(det)
def monos(deg):
    out = [sp.Integer(1)]
    for k in range(1, deg+1):
        for idx in combinations_with_replacement((q, p, y), k): out.append(sp.Mul(*idx))
    return out
random.seed(3)
pts = []
for _ in range(90):
    zv = [sp.Rational(random.randint(-12, 12), random.randint(1, 9)) for _ in range(3)]
    pts.append((zv,) + tuple(kahan_and_det(zv)))
for deg in [2, 4, 6]:
    ms = monos(deg)
    rows = []
    for zv, zn, det in pts[:len(ms) + 25]:
        s0 = dict(zip((q, p, y), zv)); s1 = dict(zip((q, p, y), zn))
        rows.append([m.subs(s1) - det*m.subs(s0) for m in ms])
    A = sp.Matrix(rows)
    ns = A.nullspace()
    print(f"degree <= {deg}: exact solutions: {len(ns)}")
    if ns:
        for v in ns:
            v = v/ v[0] if v[0] != 0 else v
            print("   P =", sp.factor(sum(c*m for c, m in zip(v, ms))))
        break
