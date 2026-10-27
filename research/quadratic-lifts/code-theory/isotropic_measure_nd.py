"""Invariant density of the lifted isotropic quartic Kahan map in R^n (state dimension 2n+1), exact search mod a prime,
and a test of the guess  P = (1 + h^2 U'(y)/2)^(n-1) (1 + h^2 u2 |q|^2 - h^4 u2 (|p|^2/2 + U(y) + u1^2/(2 u2))/2)
(for n = 2, u2 = 1/2, u1 = 0 this is the measure of the applications note)."""
import sympy as sp, numpy as np, time
from darboux_modp import density_nullity
from qlift import kahan_step
for n in (2, 3):
    q = sp.symbols(f'q1:{n+1}'); p = sp.symbols(f'p1:{n+1}'); y = sp.Symbol('y')
    F = list(p) + [-y*qi for qi in q] + [2*sum(a*b for a, b in zip(q, p))]
    for d in (2, 4, 5, 6):
        t = time.time(); nul, nm = density_nullity(F, list(q) + list(p) + [y], d)
        print(f"isotropic quartic n={n} ({2*n+1}D): densities of degree <= {d}: {nul} ({nm} monomials, {time.time()-t:.0f}s)", flush=True)
        if nul: break
# numerical test of the guessed product formula for several n, u2, u1
for (n, u2, u1, h) in [(2, .5, 0, .4), (3, .5, 0, .4), (3, 1.2, .5, .3), (4, .8, -.3, .3)]:
    def Fn(z):
        qq, pp, yy = z[:n], z[n:2*n], z[2*n]; return np.concatenate([pp, -2*(u2*yy + u1)*qq, [2*qq@pp]])
    N = 2*n + 1; I = np.eye(N); JF = lambda z: np.array([(Fn(z + I[k]) - Fn(z - I[k]))/2 for k in range(N)]).T
    def Pd(z):
        qq, pp, yy = z[:n], z[n:2*n], z[2*n]; Up = u2*yy + u1
        return (1 + h**2*Up/2)**(n - 1)*(1 + h**2*u2*(qq@qq) - h**4*u2/2*(pp@pp/2 + u2*yy**2/2 + u1*yy + u1**2/(2*u2)))
    rng = np.random.default_rng(0); worst = 0
    for _ in range(200):
        z = rng.uniform(-1, 1, N); Mm = np.eye(N) - h/2*JF(z); zn = z + h*np.linalg.solve(Mm, Fn(z))
        det = np.linalg.det(np.linalg.solve(Mm, np.eye(N) + h/2*JF(zn))); worst = max(worst, abs(Pd(zn) - det*Pd(z))/abs(Pd(z)))
    print(f"guess P = (1 + h^2 U'/2)^(n-1) * K, n={n}, u2={u2}, u1={u1}: max defect {worst:.1e}")
