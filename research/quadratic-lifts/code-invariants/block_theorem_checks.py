"""Numerical checks of the block theorem.

A block w of the state z moves as  w' = A(z) L(w)  with L a constant-coefficient affine field and A an affine
scalar. For a basis J_1..J_r of quadratic invariants of L, with T J := L(w)^T Hess(J) L(w) = sum_l Tmat[k,l] J_l,
Kahan's method conserves the vector  (Id + h^2 A(z)^2/8 * Tmat)^{-1} (J_1(w), ..., J_r(w))."""
import numpy as np
from qlift import kahan_step


def jac(F, N):
    I = np.eye(N)
    return lambda z: np.array([(F(z + I[k]) - F(z - I[k]))/2 for k in range(N)]).T   # exact for quadratic F


def max_drift(F, z, h, invariant, nsteps=300):
    JF = jac(F, len(z)); G0 = invariant(z); dev = 0
    for k in range(nsteps):
        z = kahan_step(F, JF, z, h)
        dev = max(dev, np.max(abs(invariant(z) - G0)/(1 + abs(G0))))
        if np.max(abs(z)) > 1e3: break
    return dev, k + 1


h = 0.1
# ---- Test 1: block in R^4 rotating with complex structure J2 (+) J2 at speed A (A depends on the block and on outside
# variables). All four Hermitian forms are invariants of L with T = 2 Id, so each is divided by (1 + h^2 A^2/4).
def F1(z):
    w = z[:4]; a, b = z[4], z[5]; A = a + 0.3*w[0] + 0.2*b
    Jw = np.array([-w[1], w[0], -w[3], w[2]])
    return np.concatenate([A*Jw, [-b + 0.1*w[0]*w[2], a - 0.2*w[1]**2 + 0.05*a*b]])
A1 = lambda z: z[4] + 0.3*z[0] + 0.2*z[5]
herm = [lambda z: z[0]**2 + z[1]**2, lambda z: z[2]**2 + z[3]**2, lambda z: z[0]*z[2] + z[1]*z[3], lambda z: z[0]*z[3] - z[1]*z[2]]
G1 = lambda z: np.array([f(z) for f in herm])/(1 + h**2*A1(z)**2/4)
dev, n = max_drift(F1, np.array([0.3, -0.2, 0.5, 0.1, 0.2, -0.4]), h, G1)
print(f"Test 1 (Hermitian forms / (1 + h^2 A^2/4)): max relative drift {dev:.1e} over {n} steps")

# ---- Test 2: resonant, NON-diagonalisable block: u' = A (om Ju + v), v' = A om Jv  (J = rotation generator).
# Invariants of L: G = u^T J v (rank-4 Hessian, not two-variable in any coordinates) and N = |v|^2.
# T G = 2 om^2 G - 2 om N,  T N = 2 om^2 N   ->   N~ = N/(1+2 al om^2),  G~ = (G + 2 al om N~)/(1+2 al om^2),  al = h^2 A^2/8
om = 1.3
Jr = np.array([[0., -1.], [1., 0.]])
def F2(z):
    u, v = z[:2], z[2:4]; a, b = z[4], z[5]; A = a + 0.3*u[0] + 0.2*b
    return np.concatenate([A*(om*Jr@u + v), A*om*(Jr@v), [-b + 0.1*u[0]*v[1], a - 0.2*u[1]**2 + 0.05*a*b]])
A2 = lambda z: z[4] + 0.3*z[0] + 0.2*z[5]
def G2(z, with_correction=True):
    u, v = z[:2], z[2:4]; al = h**2*A2(z)**2/8
    G, N = u@Jr@v, v@v
    Nt = N/(1 + 2*al*om**2)
    Gt = (G + (2*al*om*Nt if with_correction else 0))/(1 + 2*al*om**2)
    return np.array([Gt, Nt])
z2 = np.array([0.4, -0.3, 0.2, 0.5, 0.3, -0.2])
dev, n = max_drift(F2, z2, h, G2)
print(f"Test 2 (resonant Jordan block, G~ and N~ with squared denominator): max relative drift {dev:.1e} over {n} steps")
dev, n = max_drift(F2, z2, h, lambda z: G2(z, with_correction=False))
print(f"         same without the N-correction term (two-variable-style guess): max relative drift {dev:.1e}")

# ---- Test 3: nilpotent block w' = A (w2, w3, 0): invariant w2^2 - 2 w1 w3 (rank-3 Hessian), T I = 2 w3^2, T w3^2 = 0
# -> I~ = w2^2 - 2 w1 w3 - (h^2 A^2/4) w3^2.   Speed A = b with a' = -b, so the block's displacement is bounded and
# the nonlinear oscillator (a, b) feels the block through b' = a - 0.05 a^2 - 0.02 w2 w3: the whole orbit stays bounded.
def F3(z):
    w = z[:3]; a, b = z[3], z[4]; A = b
    return np.concatenate([A*np.array([w[1], w[2], 0.0]), [-b, a - 0.05*a**2 - 0.02*w[1]*w[2]]])
A3 = lambda z: z[4]
G3 = lambda z: np.array([z[1]**2 - 2*z[0]*z[2] - h**2*A3(z)**2/4*z[2]**2])
z3 = np.array([0.3, -0.4, 0.6, 0.2, 0.1])
dev, n = max_drift(F3, z3, h, G3, nsteps=20000)
print(f"Test 3 (nilpotent block, w2^2 - 2 w1 w3 - h^2 A^2 w3^2/4): max relative drift {dev:.1e} over {n} steps")
dev, n = max_drift(F3, z3, h, lambda z: np.array([z[1]**2 - 2*z[0]*z[2]]), nsteps=20000)
print(f"         unmodified w2^2 - 2 w1 w3: max relative drift {dev:.1e}")

# ---- Test 2 is genuinely beyond mechanisms A and B: G is an integral, but not in the span of two-variable and
# lemma-protected integrals (automatic test from protected_span.py).
from protected_span import analyse
S_G = np.zeros((6, 6)); S_G[:2, 2:4] = Jr; S_G[2:4, :2] = Jr.T
S_N = np.zeros((6, 6)); S_N[2:4, 2:4] = 2*np.eye(2)
analyse("resonant block of Test 2", F2, 6, {"G = u^T J v": (S_G, np.zeros(6)), "N = |v|^2": (S_N, np.zeros(6))})
