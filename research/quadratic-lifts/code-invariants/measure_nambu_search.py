"""Conjecture: if C is a quadratic depending on two variables and H is ANY quadratic on R^3, Kahan's map of the Nambu
field z' = grad H x grad C preserves a measure dz/P with P a polynomial of degree <= 4.
Every lift y = q^2 of a 1-DOF Hamiltonian p^2/2 + V(q) with V quartic is of this form, with C = y - q^2.
Exact discrete-Darboux-polynomial search (P(z') = det(DPhi) P(z), linear algebra over the rationals) at rational h."""
import sympy as sp, random
from itertools import combinations_with_replacement
q, p, y = sp.symbols('q p y'); z = sp.Matrix([q, p, y])
ms = [sp.Integer(1)]
for k in range(1, 5):
    for idx in combinations_with_replacement((q, p, y), k): ms.append(sp.Mul(*idx))

def densities(H, C, h, seed=5):
    """Basis of polynomial solutions P (degree <= 4) of P(z') = det(DPhi(z)) P(z)."""
    gH = sp.Matrix([sp.diff(H, v) for v in (q, p, y)]); gC = sp.Matrix([sp.diff(C, v) for v in (q, p, y)])
    F = gH.cross(gC).applyfunc(sp.expand); J = F.jacobian(z)
    random.seed(seed); rows = []
    while len(rows) < len(ms) + 20:
        zv = [sp.Rational(random.randint(-12, 12), random.randint(1, 9)) for _ in range(3)]
        sub = dict(zip((q, p, y), zv)); M = sp.eye(3) - h/2*J.subs(sub)
        if M.det() == 0: continue
        zn = sp.Matrix(zv) + h*M.LUsolve(F.subs(sub)); Jn = J.subs(dict(zip((q, p, y), list(zn))))
        det = (M.inv()*(sp.eye(3) + h/2*Jn)).det(); s1 = dict(zip((q, p, y), list(zn)))
        rows.append([m.subs(s1) - det*m.subs(sub) for m in ms])
    return [sp.factor(sum(c*m for c, m in zip(v, ms))) for v in sp.Matrix(rows).nullspace()]

R = sp.Rational; C = y - q**2; h = R(1, 5)
print("lifts y = q^2 of p^2/2 + V(q), lifted energy p^2/2 + a y^2 + b q y + c y + d q  (h = 1/5):")
for a, b, c, d in [(R(1, 4), R(1, 3), R(1, 2), 0), (R(1, 4), 1, R(-1, 2), R(1, 3)), (2, R(-1, 2), R(3, 2), -1), (R(1, 4), 0, 0, R(1, 2))]:
    sols = densities(p**2/2 + a*y**2 + b*q*y + c*y + d*q, C, h)
    print(f"   V = {a}q^4 + {b}q^3 + {c}q^2 + {d}q: {len(sols)} density/densities of degree <= 4; P = {sols[0] if sols else '-'}")
print("random quadratic H, C = y - q^2:")
rnd = random.Random(7); mons2 = [q**2, p**2, y**2, q*p, q*y, p*y, q, p, y]
for trial in range(3):
    H = sum(R(rnd.randint(-5, 5), rnd.randint(1, 4))*m for m in mons2)
    print(f"   H = {H}: {len(densities(H, C, h))} density/densities of degree <= 4 at h = 1/5,",
          f"{len(densities(H, C, R(3, 4), seed=9))} at h = 3/4")
