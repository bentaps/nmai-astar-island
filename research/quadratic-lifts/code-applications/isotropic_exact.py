"""Exact check, with h SYMBOLIC, that Kahan's map of the lifted isotropic quartic oscillator conserves c_h, M_h, E_h
and preserves the measure dz/P (identities between rational functions of h, at random rational points z)."""
import sympy as sp, random
q1, q2, p1, p2, y, h = sp.symbols('q1 q2 p1 p2 y h'); V = [q1, q2, p1, p2, y]
F = sp.Matrix([p1, p2, -q1*y, -q2*y, 2*(q1*p1 + q2*p2)]); J = F.jacobian(V)
Q2, P2, M = q1**2 + q2**2, p1**2 + p2**2, q1*p2 - q2*p1
Mh = M/(1 + h**2*y/4)
inv = {"c_h": y - Q2 + h**2*P2/4, "M_h": Mh,
       "E_h": (P2/2 + y**2/4 + h**2*Mh**2/4)/(1 + h**2*Q2/2 + h**6*Mh**2/16)}
P = (1 + h**2*y/4)*(1 + h**2*Q2/2 - h**4/4*(P2/2 + y**2/4))
random.seed(4); ok = {k: True for k in inv}; okP = True
for trial in range(4):
    zv = [sp.Rational(random.randint(-7, 7), random.randint(1, 4)) for _ in range(5)]
    sub = dict(zip(V, zv)); A = sp.eye(5) - h/2*J.subs(sub)
    zn = (sp.Matrix(zv) + h*A.LUsolve(F.subs(sub))).applyfunc(sp.cancel); subn = dict(zip(V, list(zn)))
    for k, f in inv.items(): ok[k] &= sp.cancel(f.subs(subn) - f.subs(sub)) == 0
    det = sp.cancel((sp.eye(5) + h/2*J.subs(subn)).det()/A.det())
    okP &= sp.cancel(P.subs(subn) - det*P.subs(sub)) == 0
print("conserved exactly for all h at 4 random rational points:", ok, "; measure dz/P preserved exactly:", okP)
