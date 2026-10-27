"""Paper's example: H = x1^2/2 + x2^4 + x1^2 x2^2,  x1' = -2 x1^2 x2 - 4 x2^3,  x2' = 2 x1 x2^2 + x1.
Lie-Poisson (quadratic-RHS) lift z = (x1, x2, y11, y12, y22):
  x1' = -2 x2 y11 - 4 x2 y22,      x2' = 2 x1 y22 + x1,
  y11' = -4 y11 y12 - 8 y12 y22,   y12' = y11 - 4 y22^2,   y22' = 4 y12 y22 + 2 y12
Constraint factors x1, x2 have QUADRATIC dynamics -> no exact modified constraints expected for Kahan."""
import numpy as np
from qlift import kahan_step, run
H = lambda x1, x2: x1**2/2 + x2**4 + x1**2*x2**2
def F(z):
    x1, x2, a, b, c = z
    return np.array([-2*x2*a - 4*x2*c, 2*x1*c + x1, -4*a*b - 8*b*c, a - 4*c**2, 4*b*c + 2*b])
def JF(z):
    x1, x2, a, b, c = z
    return np.array([[0, -2*a - 4*c, -2*x2, 0, -4*x2],
                     [2*c + 1, 0, 0, 0, 2*x1],
                     [0, 0, -4*b, -4*a - 8*c, -8*b],
                     [0, 0, 1, 0, -8*c],
                     [0, 0, 0, 4*c + 2, 4*b]])
lift = lambda x1, x2: np.array([x1, x2, x1**2, x1*x2, x2**2])
h = 0.05; x0 = (1.0, 0.0); H0 = H(*x0); n = 40000
# (1) Kahan on the lifted system, no reset
Z = run(lambda z: kahan_step(F, JF, z, h), lift(*x0), n)
cons = np.max(np.abs(Z[:, 2:] - np.column_stack([Z[:, 0]**2, Z[:, 0]*Z[:, 1], Z[:, 1]**2])), axis=1)
E = H(Z[:, 0], Z[:, 1])
for frac in [0.01, 0.1, 0.5, 1.0]:
    k = int(frac*n)
    print(f"Kahan-lift (no reset) t={k*h:7.1f}: max constraint err so far {np.max(cons[:k+1]):.2e}, |H-H0| {abs(E[k]-H0):.2e}")
# (2) Kahan + reset y := phi(x) after each step  (one-step linearly implicit method on x-space)
x = np.array(x0); Ers = []
for k in range(n):
    z = kahan_step(F, JF, lift(*x), h); x = z[:2]; Ers.append(H(*x) - H0)
Ers = np.abs(Ers)
for frac in [0.01, 0.1, 0.5, 1.0]:
    k = int(frac*n)-1
    print(f"Kahan-lift + reset  t={(k+1)*h:7.1f}: |H-H0| {Ers[k]:.2e}")
