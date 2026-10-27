"""Paper's H = x1^2/2 + x2^4 + x1^2 x2^2  with q = x2, p = x1.
Reduced 'quadratic-observable' lift (Z2 quotient): y = q^2, w = p^2, u = q p  (closed sp(2) Lie-Poisson system)
   y' = 2u + 4uy,   w' = -4u(w + 2y),   u' = w - 4y^2 ,  H~ = w/2 + y w + y^2, Casimir C = y w - u^2 (=0).
CMOQ two-variable theorem (I = H~(y,w), A = 4u):  Kahan preserves (H~ - h^2 u^2)/(1 - 4 h^2 u^2)."""
import numpy as np
from qlift import kahan_step, run, algebraic_relation_spectrum
F = lambda z: np.array([2*z[2] + 4*z[2]*z[0], -4*z[2]*(z[1] + 2*z[0]), z[1] - 4*z[0]**2])
JF = lambda z: np.array([[4*z[2], 0, 2 + 4*z[0]], [-8*z[2], -4*z[2], -4*(z[1] + 2*z[0])], [-8*z[0], 1, 0]])
Ht = lambda Z: Z[..., 1]/2 + Z[..., 0]*Z[..., 1] + Z[..., 0]**2
Hmod = lambda Z, h: (Ht(Z) - h**2*Z[..., 2]**2)/(1 - 4*h**2*Z[..., 2]**2)
h, n = 0.1, 10000
print(" i  x1_0   ok   max|H(q,p)-H0| (=|H~-H~0|)   drift of modified H   max|Casimir yw-u^2|")
for i in range(0, 16):
    p0, q0 = 2 + 2*i/3, 0.0
    z0 = np.array([q0**2, p0**2, q0*p0])
    Z = np.empty((n+1, 3)); Z[0] = z0; z = z0; ok = True
    for k in range(n):
        z = kahan_step(F, JF, z, h); Z[k+1] = z
        if not np.all(np.isfinite(z)) or np.max(abs(z)) > 1e8: ok = False; break
    Z = Z[:k+2]
    dH = np.max(abs(Ht(Z) - Ht(z0))); dHm = np.max(abs(Hmod(Z, h) - Hmod(z0, h)))
    cas = np.max(abs(Z[:, 0]*Z[:, 1] - Z[:, 2]**2))
    print(f"{i:2d} {p0:5.2f}  {'yes' if ok else 'NO '}  {dH:10.2e}                 {dHm:9.1e}             {cas:9.2e}")
# is the reduced Kahan map integrable (second algebraic relation)?
z0 = np.array([0.0, 4.0, 0.0]); Z = run(lambda z: kahan_step(F, JF, z, 0.05), z0, 3000)
for deg in [2, 3, 4]:
    s, _ = algebraic_relation_spectrum(Z, deg)
    print(f"deg {deg}: #monomials {len(s)}, smallest 4 sing. vals {s[-4:]}")
