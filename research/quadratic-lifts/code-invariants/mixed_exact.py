"""Exact check: Kahan on q' = qy + p, p' = -py - q, y' = p^2 - q^2 conserves
   Ia_h = (Ia - h^2 u^2/4)/(1 + h^2 u^2/2),  Ia = y + v^2 + y^2/2   (= c + Ht, two variables (v,y), A = u)
   Ib_h = (Ib - h^2 v^2/4)/(1 + h^2 v^2/2),  Ib = u^2 + y^2/2 - y   (= Ht - c, two variables (u,y), A = v)
with u = (q+p)/sqrt2, v = (q-p)/sqrt2, so the constraint c = y - qp = (Ia - Ib)/2 has the modified version (Ia_h - Ib_h)/2."""
import sympy as sp
from fractions import Fraction as Fr
import random
q, p, y, h = sp.symbols('q p y h')
z = sp.Matrix([q, p, y]); F = sp.Matrix([q*y + p, -p*y - q, p**2 - q**2]); J = F.jacobian(z)
u2 = lambda Z: (Z[0] + Z[1])**2/2; v2 = lambda Z: (Z[0] - Z[1])**2/2
Ia = lambda Z, hv: (Z[2] + v2(Z) + Z[2]**2/2 - hv**2*u2(Z)/4)/(1 + hv**2*u2(Z)/2)
Ib = lambda Z, hv: (u2(Z) + Z[2]**2/2 - Z[2] - hv**2*v2(Z)/4)/(1 + hv**2*v2(Z)/2)
random.seed(0); ok = True
for t in range(10):
    hv = sp.Rational(random.randint(1, 9), random.randint(3, 13))
    zv = [sp.Rational(random.randint(-20, 20), random.randint(1, 11)) for _ in range(3)]
    sub = {q: zv[0], p: zv[1], y: zv[2], h: hv}
    zn = sp.Matrix(zv) + hv*((sp.eye(3) - hv/2*J.subs(sub)).LUsolve(F.subs(sub)))
    ok &= sp.simplify(Ia(zn, hv) - Ia(zv, hv)) == 0 and sp.simplify(Ib(zn, hv) - Ib(zv, hv)) == 0
print("both predicted modified invariants exact at 10 random rational points/steps:", ok)
