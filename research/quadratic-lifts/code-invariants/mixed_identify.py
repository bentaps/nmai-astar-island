import numpy as np, sympy as sp
from qlift import kahan_step, run, monomial_matrix
F = lambda z: np.array([z[0]*z[2] + z[1], -z[1]*z[2] - z[0], z[1]**2 - z[0]**2])
JF = lambda z: np.array([[z[2], 1, z[0]], [-1, -z[2], -z[1]], [-2*z[0], 2*z[1], 0]])
h = 0.2
allv = []
rng = np.random.default_rng(3)
for t in range(10):
    z0 = rng.uniform(-0.8, 0.8, 3)
    Z = run(lambda z: kahan_step(F, JF, z, h), z0, 2500)
    A, names = monomial_matrix(Z, 2)
    U, s, Vt = np.linalg.svd(A, full_matrices=False); s = s/s[0]
    allv.append(Vt[s < 1e-11])
allv = np.vstack(allv)
U, s, Vt = np.linalg.svd(allv); r = int(np.sum(s > 1e-8))
print("span of quadratic relations across orbits has dim", r, " monomials:", [''.join('qpy'[i] for i in n) or '1' for n in names])
M = sp.Matrix(np.round(Vt[:r], 12)); R, piv = M.rref(iszerofunc=lambda x: abs(x) < 1e-9)
for i in range(R.rows):
    row = [float(R[i, j]) for j in range(R.cols)]
    print({(''.join('qpy'[k] for k in names[j]) or '1'): round(row[j], 8) for j in range(len(row)) if abs(row[j]) > 1e-9})
print("h^2 =", h**2, " h^2/4 =", h**2/4, " h^2/8 =", h**2/8)
