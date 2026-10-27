"""Isotropic oscillator H = |p|^2/2 + U(|q|^2), U(y) = u2 y^2/2 + u1 y, in R^n, lifted with y = |q|^2.
Modified energy (u1 enters only through the shift y -> y + u1/u2, an affine change of variables):
   E_h = (|p|^2/2 + U(y) + u1^2/(2 u2) + alpha |M_h|^2) / (1 + h^2 u2 |q|^2 + gamma |M_h|^2),
   M_h = M/(1 + h^2 U'(y)/2),  |M|^2 = sum_{a<b} (q_a p_b - q_b p_a)^2,  alpha = h^2 u2/2,  gamma = h^6 u2^2/4."""
import numpy as np, sys
from qlift import kahan_step, run

def check(n, u2, u1, h, alpha=None, gamma=None, nsteps=20000, seed=1):
    alpha = h**2*u2/2 if alpha is None else alpha; gamma = h**6*u2**2/4 if gamma is None else gamma
    def F(z):
        q, p, y = z[:n], z[n:2*n], z[2*n]; return np.concatenate([p, -2*(u2*y + u1)*q, [2*q@p]])
    I = np.eye(2*n + 1); JF = lambda z: np.array([(F(z + I[k]) - F(z - I[k]))/2 for k in range(2*n + 1)]).T
    rng = np.random.default_rng(seed); q0 = rng.normal(size=n)*0.7; p0 = rng.normal(size=n)*0.5
    Z = run(lambda z: kahan_step(F, JF, z, h), np.concatenate([q0, p0, [q0@q0 - h**2/4*p0@p0]]), nsteps)
    q, p, y = Z[:, :n], Z[:, n:2*n], Z[:, 2*n]
    M2 = sum((q[:, a]*p[:, b] - q[:, b]*p[:, a])**2 for a in range(n) for b in range(a + 1, n))
    Mh2 = M2/(1 + h**2*(u2*y + u1)/2)**2
    Q2, P2 = (q**2).sum(1), (p**2).sum(1)
    Eh = (P2/2 + u2*y**2/2 + u1*y + u1**2/(2*u2) + alpha*Mh2)/(1 + h**2*u2*Q2 + gamma*Mh2)
    return abs(Eh - Eh[0]).max()/abs(Eh[0]), abs(Mh2 - Mh2[0]).max()/Mh2[0]

for (n, u2, u1, h) in [(2, .5, 0, .4), (3, .5, 0, .4), (4, .5, 0, .4), (2, 1.0, 0, .4), (3, 1.3, 0, .3), (2, .5, .7, .4), (3, 1.0, -.4, .3)]:
    dE, dM = check(n, u2, u1, h)
    print(f"n={n}, u2={u2}, u1={u1}, h={h}: drift of E_h {dE:.1e};  of |M_h|^2 {dM:.1e}")
