import numpy as np
from qlift import kahan_step, run, monomial_matrix

F = lambda z: np.array([z[1], -z[0]*z[2], 2*z[0]*z[1]])
JF = lambda z: np.array([[0, 1, 0], [-z[2], 0, -z[0]], [2*z[1], 2*z[0], 0]])
np.set_printoptions(precision=6, suppress=True, linewidth=150)

h = 0.2
def nullvecs(Z, deg=2, tol=1e-11):
    A, names = monomial_matrix(Z, deg)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = s/s[0]
    return Vt[s < tol], names

# collect null vectors from several orbits
allv = []
for (q0, p0, y0) in [(1.0, 0.5, 1.0), (0.3, 1.2, 0.09), (1.5, -0.2, 2.0), (0.7, 0.7, 0.2), (-1.1, 0.4, 1.5), (0.2, 0.1, -0.3)]:
    Z = run(lambda z: kahan_step(F, JF, z, h), np.array([q0, p0, y0]), 3000)
    V, names = nullvecs(Z)
    print("orbit", (q0, p0, y0), "null dim", len(V))
    allv.append(V)
allv = np.vstack(allv)
print("monomials:", names)
# the span of all null vectors across orbits: (1, q, p, y, q^2, qp, qy, p^2, py, y^2)
U, s, Vt = np.linalg.svd(allv)
print("singular values of stacked null vectors:", s)
basis = Vt[:np.sum(s > 1e-8)]
# express basis in a reduced echelon-like form for readability
import sympy as sp
M = sp.Matrix(np.round(basis, 12))
R, piv = M.rref(iszerofunc=lambda x: abs(x) < 1e-9)
print("pivot columns:", [names[i] for i in piv])
for i in range(R.rows):
    row = [float(R[i, j]) for j in range(R.cols)]
    if max(abs(np.array(row))) > 1e-9:
        print({str(names[j]): round(row[j], 8) for j in range(len(row)) if abs(row[j]) > 1e-9})
