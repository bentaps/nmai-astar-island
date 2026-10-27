"""Invariant measure of the Kahan map for the trig-lifted pendulum (s, c, p):
search for P (deg <= 4) with P(z') = det(DPhi(z)) P(z), using DPhi = (I - h/2 F'(z))^{-1} (I + h/2 F'(z')).
Result: P = 1 + h^2 p^2/4 (and s^2 + c^2, equivalent via K_h), i.e. the measure ds dc dp / (1 + h^2 p^2/4)."""
import numpy as np
from qlift import monomial_matrix
h = 0.25
F = lambda z: np.array([z[1]*z[2], -z[0]*z[2], -z[0]])
JF = lambda z: np.array([[0, z[2], z[1]], [-z[2], 0, -z[0]], [-1, 0, 0]])
def step(z):
    M = np.eye(3) - h/2*JF(z); zn = z + h*np.linalg.solve(M, F(z))
    return zn, np.linalg.det(np.linalg.solve(M, np.eye(3) + h/2*JF(zn)))
rng = np.random.default_rng(0); Z = rng.uniform(-1.2, 1.2, size=(600, 3))
R = [step(z) for z in Z]; Zn = np.array([r[0] for r in R]); d = np.array([r[1] for r in R])
for deg in [1, 2, 3, 4]:
    A0, names = monomial_matrix(Z, deg); A1, _ = monomial_matrix(Zn, deg)
    U, sv, Vt = np.linalg.svd(A1 - d[:, None]*A0); sv = sv/sv[0]; k = np.sum(sv < 1e-10)
    print(f"deg {deg}: null dim {k}, smallest sv {sv[-1]:.1e}")
    if k:
        for v in Vt[-k:]:
            v = v/v[np.argmax(abs(v))]
            print("   P =", " + ".join(f"{x:.5g}*{''.join('scp'[i] for i in nm) or '1'}" for x, nm in zip(v, names) if abs(x) > 1e-8))
        print(f"   (h^2/4 = {h**2/4})")
        break
