"""Can polarisation (z = x_i + x_j, then x_i x_j = (z^2 - x_i^2 - x_j^2)/2) make every constraint Kahan-friendly? No.
Test on the cubic example H = p^2/2 + q^2 p^2 + q^4 of the MQAV paper, lifted with z = q + p, a = q^2, b = p^2, w = z^2,
qp := (w - a - b)/2. The linear constraint is conserved, but the squares are not exact integrals (they cannot be,
because d/dt q^2 = 2 q q' is cubic unless q' is affine), so they drift and here Kahan's method blows up."""
import numpy as np
from qlift import kahan_step, run
from protected_span import analyse

def Fpol(v):
    q, p, z, a, b, w = v; m = (w - a - b)/2
    qd = p + 2*a*p; pd = -2*b*q - 4*a*q
    return np.array([qd, pd, qd + pd, 2*m*(1 + 2*a), -4*m*(b + 2*a),
                     2*(m + b) + 4*a*(m + b) - 4*b*(a + m) - 8*a*(a + m)])
def quad(N, pairs):
    S = np.zeros((N, N)); s = np.zeros(N)
    for (i, j, c) in pairs:
        if j is None: s[i] += c
        elif i == j: S[i, i] += 2*c
        else: S[i, j] += c; S[j, i] += c
    return S, s
analyse("polarized lift of H = p^2/2 + q^2 p^2 + q^4", Fpol, 6,
        {"linear  z - q - p": quad(6, [(2, None, 1), (0, None, -1), (1, None, -1)]),
         "square  a - q^2": quad(6, [(3, None, 1), (0, 0, -1)]), "square  b - p^2": quad(6, [(4, None, 1), (1, 1, -1)]),
         "square  w - z^2": quad(6, [(5, None, 1), (2, 2, -1)]),
         "product (w-a-b)/2 - qp": quad(6, [(5, None, .5), (3, None, -.5), (4, None, -.5), (0, 1, -1)])})
I6 = np.eye(6); JF = lambda v: np.array([(Fpol(v + I6[k]) - Fpol(v - I6[k]))/2 for k in range(6)]).T
q0, p0, h = 0.6, 0.3, 0.05
V = run(lambda v: kahan_step(Fpol, JF, v, h), np.array([q0, p0, q0 + p0, q0**2, p0**2, (q0 + p0)**2]), 20000)
q, p, z, a, b, w = V.T; k = len(q)//10
for lab, c in [("z - q - p", z - q - p), ("a - q^2", a - q**2), ("w - z^2", w - z**2), ("(w-a-b)/2 - qp", (w - a - b)/2 - q*p)]:
    print(f"Kahan, h={h}: max|{lab:15s}| in the first / last 10% of 2e4 steps: {abs(c[:k]).max():.1e} / {abs(c[-k:]).max():.1e}")
