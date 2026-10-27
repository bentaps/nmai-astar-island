import numpy as np
from qlift import kahan_step, run

F = lambda z: np.array([z[1], -z[0]*z[2], 2*z[0]*z[1]])
JF = lambda z: np.array([[0, 1, 0], [-z[2], 0, -z[0]], [2*z[1], 2*z[0], 0]])
np.set_printoptions(precision=8, suppress=True, linewidth=150)
h = 0.2
G1 = lambda Z: Z[:, 2] - h**2/8*Z[:, 2]**2
G3 = lambda Z: Z[:, 1]**2 + 0.5*Z[:, 2]**2
# On each orbit find (a, b, d) with a + b*G1 + d*G3 = 0
for (q0, p0, y0) in [(1.0, 0.5, 1.0), (0.3, 1.2, 0.09), (1.5, -0.2, 2.0), (0.7, 0.7, 0.2), (-1.1, 0.4, 1.5), (0.2, 0.1, -0.3)]:
    Z = run(lambda z: kahan_step(F, JF, z, h), np.array([q0, p0, y0]), 2000)
    A = np.column_stack([np.ones(len(Z)), G1(Z), G3(Z)])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    v = Vt[-1]; v = v/v[2]
    print(f"orbit {(q0,p0,y0)}: smallest sv {s[-1]/s[0]:.2e};  relation 1*{v[0]:.8f} + G1*{v[1]:.8f} + G3 = 0")
