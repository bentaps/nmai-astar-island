"""Generalised Suslov system in R^4: x' = l(x) J Hess x, H = x' Hess x / 2 (two normal modes, frequencies w1, w2).
Block theorem prediction: Kahan conserves each normal-mode action I_k / (1 + h^2 l(x)^2 w_k^2 / 4)."""
import numpy as np
from qlift import kahan_step, run
rng = np.random.default_rng(2)
J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
A = rng.normal(size=(4, 4)); Hs = A@A.T + np.eye(4)          # positive definite Hessian
l = rng.normal(size=4)*0.5
M = J@Hs
F = lambda x: (l@x)*(M@x)
JF = lambda x: np.outer(M@x, l) + (l@x)*M
# normal modes of the linear Hamiltonian system x' = M x: actions via the symplectic eigenbasis
ev, V = np.linalg.eig(M)
w = np.sort(np.abs(ev.imag))[::2]
acts = []
for k, wk in enumerate(np.unique(np.round(np.abs(ev.imag), 12))):
    idx = [i for i in range(4) if abs(abs(ev[i].imag) - wk) < 1e-9]
    v = V[:, idx[0]]
    # quadratic invariant of the mode: |<conj eigvec of M^T, x>|^2  (left eigenvector projection)
    evT, U = np.linalg.eig(M.T)
    j = np.argmin(abs(evT - ev[idx[0]]))
    u = U[:, j]
    acts.append((wk, u))
h = 0.1; x0 = rng.normal(size=4)*0.6
X = run(lambda x: kahan_step(F, JF, x, h), x0, 20000)
for wk, u in acts:
    Ik = np.abs(X@u)**2
    Gk = Ik/(1 + h**2*(X@l)**2*wk**2/4)
    print(f"mode w={wk:.4f}: drift of I_k/(1 + h^2 l^2 w_k^2/4) = {np.max(abs(Gk - Gk[0]))/Gk[0]:.1e};  drift of plain I_k = {np.max(abs(Ik - Ik[0]))/Ik[0]:.1e}")
