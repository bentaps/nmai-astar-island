"""Kahan's method for the open Toda lattice in the variables (p, u), u_k = exp(q_k - q_{k+1}):
p_k' = u_{k-1} - u_k,  u_k' = u_k (p_k - p_{k+1}).
1. It preserves the EXACT Liouville measure dp du/(u_1 ... u_{n-1}) for every h. Reason: in (p, x = log u) one Kahan
   step is a composition of three shears, x -> x - log(1 - h Dp/2), then p -> p + (function of the shifted x),
   then x -> x + log(1 + h Dp'/2), each of which preserves volume.
2. It conserves the modified energy E - (h^2/8)|p'|^2 (companion lemma).
3. It is NOT Poisson for the log-canonical bracket, and (see integrability_degrees.py) NOT integrable for n = 3."""
import numpy as np
from darboux_modp import density_nullity
import sympy as sp

def make(n):
    def Ft(z):
        p, u = z[:n], z[n:]; pd = np.zeros(n); pd[:-1] -= u; pd[1:] += u
        return np.concatenate([pd, u*(p[:-1] - p[1:])])
    N = 2*n - 1; I = np.eye(N)
    return Ft, (lambda z: np.array([(Ft(z + I[k]) - Ft(z - I[k]))/2 for k in range(N)]).T), N

for n in (3, 4, 5):
    Ft, JF, N = make(n); rng = np.random.default_rng(0); dmeas = dpois = 0
    def K(z):
        u = z[n:]; M = np.zeros((N, N))
        for k in range(n - 1):
            M[n + k, k] = u[k]; M[n + k, k + 1] = -u[k]; M[k, n + k] = -u[k]; M[k + 1, n + k] = u[k]
        return M
    for _ in range(50):
        z = np.concatenate([rng.normal(size=n), np.exp(rng.normal(size=n - 1)*0.5)]); h = 0.4
        M = np.eye(N) - h/2*JF(z); zn = z + h*np.linalg.solve(M, Ft(z)); D = np.linalg.solve(M, np.eye(N) + h/2*JF(zn))
        dmeas = max(dmeas, abs(np.linalg.det(D)*np.prod(z[n:])/np.prod(zn[n:]) - 1))
        dpois = max(dpois, np.abs(D@K(z)@D.T - K(zn)).max()/np.abs(K(zn)).max())
    P = sp.symbols(f'p1:{n+1}'); U = sp.symbols(f'u1:{n}')
    F = [(-U[0] if k == 0 else (U[k-1] - U[k] if k < n - 1 else U[n-2])) for k in range(n)] + [U[k]*(P[k] - P[k+1]) for k in range(n - 1)]
    nul, nm = density_nullity(F, list(P) + list(U), n - 1)
    print(f"open Toda n={n}: Liouville-measure defect {dmeas:.1e}; polynomial densities of degree <= {n-1}: {nul};"
          f" Poisson (log-canonical) defect {dpois:.1e}")
