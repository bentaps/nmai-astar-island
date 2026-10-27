"""Discrete Darboux polynomial search: find P (deg <= d) with P(z') = det(DPhi(z)) P(z)  ->  measure dz/P(z) preserved."""
import numpy as np, sympy as sp
from qlift import monomial_matrix
q, p, y = sp.symbols('q p y'); h = sp.Rational(1, 5)
z = sp.Matrix([q, p, y]); F = sp.Matrix([p, -q*y, 2*q*p]); J = F.jacobian(z)
zn = z + h*(sp.eye(3) - h/2*J).LUsolve(F)
det = sp.simplify(zn.jacobian(z).det())
fz = sp.lambdify((q, p, y), list(zn), 'numpy'); fd = sp.lambdify((q, p, y), det, 'numpy')
rng = np.random.default_rng(0); Z = rng.uniform(-1.5, 1.5, size=(400, 3))
Zn = np.array([fz(*row) for row in Z], float); dets = np.array([fd(*row) for row in Z], float)
for d in [1, 2, 3, 4]:
    A0, names = monomial_matrix(Z, d); A1, _ = monomial_matrix(Zn, d)
    M = A1 - dets[:, None]*A0
    U, s, Vt = np.linalg.svd(M); s = s/s[0]
    k = np.sum(s < 1e-10)
    print(f"deg {d}: null dim {k}, smallest sv {s[-1]:.1e}")
    if k:
        v = Vt[-1]; v = v/v[np.argmax(abs(v))]
        print("   P =", " + ".join(f"{c:.6g}*{''.join('qpy'[i] for i in nm) or '1'}" for c, nm in zip(v, names) if abs(c) > 1e-8))
        break
