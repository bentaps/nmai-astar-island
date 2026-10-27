"""Mixed block: H = (qp)^2/2 + (q^2+p^2)/2, lift y = qp.
   q' = q y + p,  p' = -p y - q,  y' = p^2 - q^2         (quadratic field)
Exact invariants of the lifted field: c = y - q p  and  Ht = (y^2 + q^2 + p^2)/2.
The block (q,p,y) moves by hyperbolic SCALING (rate y) + TRANSLATIONS (rates p, -q): the 'mixed' class."""
import numpy as np
from qlift import kahan_step, run, algebraic_relation_spectrum
F = lambda z: np.array([z[0]*z[2] + z[1], -z[1]*z[2] - z[0], z[1]**2 - z[0]**2])
JF = lambda z: np.array([[z[2], 1, z[0]], [-1, -z[2], -z[1]], [-2*z[0], 2*z[1], 0]])
h = 0.1
for q0, p0 in [(0.6, 0.2), (1.0, 0.5)]:
    z0 = np.array([q0, p0, q0*p0])
    Z = run(lambda z: kahan_step(F, JF, z, h), z0, 4000)
    print(f"orbit from (q,p)=({q0},{p0}):")
    for deg in [2, 3, 4, 5, 6]:
        s, _ = algebraic_relation_spectrum(Z, deg)
        k = int(np.sum(s < 1e-11))
        print(f"   deg {deg}: monomials {len(s):3d}, exact relations {k:3d}, next sv {s[len(s)-k-1]:.1e}")
    Zl = run(lambda z: kahan_step(F, JF, z, h), z0, 200000)
    c = Zl[:, 2] - Zl[:, 0]*Zl[:, 1]; E = 0.5*(Zl[:, 0]*Zl[:, 1])**2 + 0.5*(Zl[:, 0]**2 + Zl[:, 1]**2)
    n = len(c)//10
    print(f"   long run 2e5 steps: max|y-qp| first/last 10%: {abs(c[:n]).max():.2e} / {abs(c[-n:]).max():.2e};"
          f"  |H-H0| first/last 10%: {abs(E[:n]-E[0]).max():.2e} / {abs(E[-n:]-E[0]).max():.2e}")
