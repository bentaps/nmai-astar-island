"""Exact rational-arithmetic check of: (i) C_h, (ii) H_h invariant, (iii) D(z') = det(DPhi) D(z), D = 1 + h^2 q^2/2."""
from fractions import Fraction as Fr
import random, sympy as sp
q, p, y, h = sp.symbols('q p y h')
z = sp.Matrix([q, p, y]); F = sp.Matrix([p, -q*y, 2*q*p]); J = F.jacobian(z)
random.seed(1); ok = True
for trial in range(12):
    hv = Fr(random.randint(1, 9), random.randint(2, 11))
    zv = [Fr(random.randint(-30, 30), random.randint(1, 17)) for _ in range(3)]
    sub = {q: zv[0], p: zv[1], y: zv[2], h: hv}
    M = (sp.eye(3) - h/2*J).subs(sub); Fv = F.subs(sub)
    zn = sp.Matrix(zv) + hv*M.LUsolve(Fv)
    # Jacobian of the map at zv: d z'/dz = (I - h/2 F'(z))^{-1} (I + h/2 F'(z'))   (exact formula for Kahan)
    Jn = J.subs({q: zn[0], p: zn[1], y: zn[2]})
    Dphi = M.inv() * (sp.eye(3) + sp.Rational(hv.numerator, hv.denominator)/2*Jn)
    det = Dphi.det()
    C = lambda Z: Z[2] - Z[0]**2 + hv**2/4*Z[1]**2
    Hh = lambda Z: (Z[1]**2/2 + Z[2]**2/4)/(1 + hv**2*Z[0]**2/2)
    D = lambda Z: 1 + hv**2*Z[0]**2/2
    r = [sp.nsimplify(C(zn) - C(zv)), sp.nsimplify(Hh(zn) - Hh(zv)), sp.nsimplify(D(zn) - det*D(zv))]
    ok &= all(x == 0 for x in r)
print("all exact identities hold at 12 random rational points/step sizes:", ok)
