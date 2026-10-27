"""Theorem check: Kahan's method in exponential / harmonic variables preserves the canonical Liouville measure.
H = 1/2 p^T M p + sum_k c_k exp(a_k . q) + sum_j g_j cos(b_j . q), lifted with u_k = exp(a_k . q) and
(c_j, s_j) = (cos, sin)(b_j . q):  u_k' = u_k a_k^T M p,  c_j' = -s_j b_j^T M p,  s_j' = c_j b_j^T M p,
p' = -sum c_k a_k u_k + sum g_j b_j s_j.  Prediction (three shears in (p, log u, log|zeta|, arg zeta)):
the density 1/P with P = prod_k u_k * prod_j (c_j^2 + s_j^2) satisfies P(z') = det(DPhi) P(z), for ANY mass matrix,
any number of exponentials/harmonics (independent or not); plus E_h = E - h^2/8 p'^T M p' and the modified circles."""
import numpy as np
rng = np.random.default_rng(0)

def make(M, A, cA, B, gB):
    n = M.shape[0]; ka, kb = len(A), len(B)
    def F(z):
        p = z[:n]; u = z[n:n+ka]; c = z[n+ka:n+ka+kb]; s = z[n+ka+kb:]; v = M@p
        pd = -(A.T@(cA*u)) + B.T@(gB*s)
        return np.concatenate([pd, u*(A@v), -s*(B@v), c*(B@v)])
    N = n + ka + 2*kb; I = np.eye(N)
    JF = lambda z: np.array([(F(z + I[k]) - F(z - I[k]))/2 for k in range(N)]).T   # exact: F is quadratic
    P = lambda z: np.prod(z[n:n+ka])*np.prod(z[n+ka:n+ka+kb]**2 + z[n+ka+kb:]**2)
    E = lambda z: 0.5*z[:n]@M@z[:n] + cA@z[n:n+ka] + gB@z[n+ka:n+ka+kb]
    return F, JF, P, E, n, N

cases = {
    "open Toda n=4 (independent exponents)": (np.eye(4), np.array([[1., -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]]), np.ones(3), np.zeros((0, 4)), np.zeros(0)),
    "periodic Toda n=3 (dependent exponents)": (np.eye(3), np.array([[1., -1, 0], [0, 1, -1], [-1, 0, 1]]), np.ones(3), np.zeros((0, 3)), np.zeros(0)),
    "coupled pendula, non-diagonal mass matrix": (np.array([[1., .4], [.4, .8]]), np.zeros((0, 2)), np.zeros(0), np.eye(2), np.array([1., .7])),
    "mixed: exp + cos, 3 DOF, random mass matrix": (None, np.array([[1., .5, -.3], [.2, -1., .4]]), np.array([.8, 1.3]), np.array([[.3, .7, 1.], [1., -.6, .2]]), np.array([1.1, -.5])),
}
if __name__ == "__main__":
  for name, (M, A, cA, B, gB) in cases.items():
      if M is None:
          X = rng.normal(size=(3, 3)); M = X@X.T + np.eye(3)
      F, JF, P, E, n, N = make(M, A, cA, B, gB); h = 0.37; defs = []; worstE = 0; skipped = 0
      for _ in range(200):
          p = rng.normal(size=n); q = rng.normal(size=n)*0.5
          r = rng.uniform(.5, 2, len(B))                      # (c, s) = r (cos, sin): generic points of the plane
          z = np.concatenate([p, np.exp(A@q)*rng.uniform(.5, 2, len(A)), np.cos(B@q)*r, np.sin(B@q)*r])
          Mm = np.eye(N) - h/2*JF(z); zn = z + h*np.linalg.solve(Mm, F(z))
          det = np.linalg.det(np.linalg.solve(Mm, np.eye(N) + h/2*JF(zn)))
          ell = np.concatenate([A@(M@p), B@(M@p)])
          if np.abs(h*ell/2).max() > 0.5: skipped += 1; continue      # huge step (|h l/2| > 1/2): rounding-dominated, skip
          # (exactness for all points is checked in rational arithmetic by exp_trig_measure_exact.py)
          defs.append(abs(P(zn) - det*P(z))/abs(P(z)))
          Eh = lambda z: E(z) - h**2/8*F(z)[:n]@M@F(z)[:n]
          worstE = max(worstE, abs(Eh(zn) - Eh(z))/(1 + abs(Eh(z))))
      print(f"{name:45s} measure defect median {np.median(defs):.0e}, max {max(defs):.0e} ({skipped} huge-step points skipped);"
            f"  modified-energy defect {worstE:.1e}")
