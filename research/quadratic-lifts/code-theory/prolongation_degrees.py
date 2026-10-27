"""Degree growth: the prolonged Euler-top Kahan map is not integrable, while plain Kahan on the Euler top is.
Also lists the exact quadratic integrals of the prolonged field: |x|^2 is not one of them."""
import sympy as sp, numpy as np, time
from degree_growth import degree_sequence
from protected_span import analyse
x1, x2, x3, v1, v2, v3 = V = sp.symbols('x1 x2 x3 v1 v2 v3'); a = [1, -2, 1]
f = sp.Matrix([a[0]*x2*x3, a[1]*x3*x1, a[2]*x1*x2]); X = sp.Matrix([x1, x2, x3]); vv = sp.Matrix([v1, v2, v3])
F = list(vv) + list(f.jacobian(X)*vv)
t = time.time(); print("prolonged Euler top (6D): degrees", degree_sequence(F, list(V), K=6, mmax=400), f"({time.time()-t:.0f}s)")
t = time.time(); print("plain Kahan Euler top (3D): degrees", degree_sequence(list(f), [x1, x2, x3], K=6, mmax=400), f"({time.time()-t:.0f}s)")
Fn = sp.lambdify([V], F, 'numpy'); Fnum = lambda z: np.array(Fn(z), float)
def quad(N, entries):
    S = np.zeros((N, N)); s = np.zeros(N)
    for (i, j, c) in entries:
        if j is None: s[i] += c
        elif i == j: S[i, i] += 2*c
        else: S[i, j] += c; S[j, i] += c
    return S, s
analyse("prolonged Euler top", Fnum, 6, {"|x|^2": quad(6, [(0, 0, 1), (1, 1, 1), (2, 2, 1)]),
                                         "v1 - a1 x2 x3": quad(6, [(3, None, 1), (1, 2, -a[0])])})
