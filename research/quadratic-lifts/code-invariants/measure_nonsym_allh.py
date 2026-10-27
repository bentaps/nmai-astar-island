"""Invariant measure of Kahan's map for the non-symmetric quartic lift, valid for EVERY step size h.
H = p^2/2 + q^4/4 + q^3/3 + q^2/2, lift y = q^2:  q' = p,  p' = -(y/3 + 2q(y/2 + q/3 + 1/2)),  y' = 2qp.
Found by an exact discrete-Darboux-polynomial search at 12 rational step sizes plus rational interpolation in h^2
(see measure_exact.py for the single-h search). Claim: with t = h^2,
  P = 432 + 72 t (1 + 2q + 2q^2 + y) + t^2 (-9 - 12q - 10y + 16q^2 + 12qy - 9y^2 - 24q^3 + 36q^2 y - 36q^4)
      + t^3 p^2 (1 + 3q)^2
satisfies P(z') = det(DPhi(z)) P(z), i.e. dq dp dy / P is preserved. Here this is checked with h kept SYMBOLIC
at random rational points z (an identity between rational functions of h)."""
import sympy as sp, random
q, p, y, h = sp.symbols('q p y h'); z = sp.Matrix([q, p, y]); t = h**2
F = sp.Matrix([p, -(y/3 + 2*q*(y/2 + q/3 + sp.Rational(1, 2))), 2*q*p]); J = F.jacobian(z)
P = (432 + 72*t*(1 + 2*q + 2*q**2 + y) + t**2*(-9 - 12*q - 10*y + 16*q**2 + 12*q*y - 9*y**2 - 24*q**3 + 36*q**2*y - 36*q**4)
     + t**3*p**2*(1 + 3*q)**2)
random.seed(11); ok = True
for trial in range(6):
    zv = [sp.Rational(random.randint(-9, 9), random.randint(1, 5)) for _ in range(3)]
    sub = dict(zip((q, p, y), zv))
    M = sp.eye(3) - h/2*J.subs(sub)
    zn = (sp.Matrix(zv) + h*M.LUsolve(F.subs(sub))).applyfunc(sp.cancel)
    Jn = J.subs(dict(zip((q, p, y), list(zn))))
    det = sp.cancel((sp.eye(3) + h/2*Jn).det()/M.det())
    lhs = P.subs(dict(zip((q, p, y), list(zn)))); rhs = det*P.subs(sub)
    ok &= sp.cancel(lhs - rhs) == 0
print("P(z') == det(DPhi) P(z) as rational functions of h at 6 random rational points:", ok)
print("on the sheet y = q^2 the h^2 term is (h^2/6) V''(q):",
      sp.simplify(sp.expand((72*(1 + 2*q + 2*q**2 + y)/432).subs(y, q**2)) - sp.diff(q**4/4 + q**3/3 + q**2/2, q, 2)/6) == 0)
