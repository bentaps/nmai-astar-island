import numpy as np
from qlift import kahan_step, midpoint_step, run, algebraic_relation_spectrum

# Lifted quartic oscillator H = p^2/2 + q^4/4, z = (q, p, y), y ~ q^2
F = lambda z: np.array([z[1], -z[0]*z[2], 2*z[0]*z[1]])
JF = lambda z: np.array([[0, 1, 0], [-z[2], 0, -z[0]], [2*z[1], 2*z[0], 0]])
H = lambda q, p: 0.5*p**2 + 0.25*q**4
Ht = lambda z: 0.5*z[..., 1]**2 + 0.25*z[..., 2]**2

h = 0.2
q0, p0 = 1.0, 0.5
n = 4000

# (a) Kahan, y0 = q0^2  (start on the true constraint manifold)
Z = run(lambda z: kahan_step(F, JF, z, h), np.array([q0, p0, q0**2]), n)
C = Z[:, 2] - Z[:, 0]**2
Ct = C + h**2/4*Z[:, 1]**2
print("Kahan, y0=q0^2:  max|C| =", np.max(abs(C)), "  max|C~ - C~0| =", np.max(abs(Ct - Ct[0])))
print("   energy error max |H(q,p)-H0| =", np.max(abs(H(Z[:,0],Z[:,1]) - H(q0,p0))),
      "  lifted |H~ - H~0| =", np.max(abs(Ht(Z)-Ht(Z[0]))))

# integrability test: does the 3D orbit lie on a curve (2 algebraic relations)?
for deg in [2, 3, 4, 5, 6]:
    s, names = algebraic_relation_spectrum(Z[:, :3], deg)
    small = np.sum(s < 1e-11)
    print(f"  deg {deg}: #monomials {len(s)}, #sing.vals < 1e-11: {small}, smallest few: {s[-small-2:] if small else s[-3:]}")
