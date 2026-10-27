"""Exact search, modulo a large prime, for polynomial invariant densities of Kahan maps.
A measure dz/P(z) is preserved iff P(z') = det(DPhi(z)) P(z) (P is a discrete Darboux polynomial whose cofactor is
the Jacobian determinant). Unknown coefficients of P (all monomials up to degree d) solve a linear system; we build it
at random points of GF(p)^N and compute its nullity by modular Gaussian elimination. Nullity > 0 means a density
exists (for a generic step size h, also taken at random mod p)."""
import numpy as np, sympy as sp, random
from itertools import combinations_with_replacement
PR = 2**31 - 1

def to_mod(r):
    r = sp.Rational(r); return (int(r.p) % PR) * pow(int(r.q) % PR, PR - 2, PR) % PR

class ModField:
    def __init__(self, F, vars):
        self.n = len(vars)
        self.F = [[(m, to_mod(c)) for m, c in sp.Poly(sp.expand(f), *vars).terms()] for f in F]
        Jm = sp.Matrix(F).jacobian(vars)
        self.J = [[[(m, to_mod(c)) for m, c in sp.Poly(Jm[i, j], *vars).terms()] if Jm[i, j] != 0 else []
                   for j in range(self.n)] for i in range(self.n)]
    @staticmethod
    def ev(terms, z):
        s = 0
        for m, c in terms:
            t = c
            for zi, e in zip(z, m):
                if e: t = t*pow(zi, e, PR) % PR
            s += t
        return s % PR
    def F_at(self, z): return [self.ev(t, z) for t in self.F]
    def J_at(self, z): return [[self.ev(t, z) for t in row] for row in self.J]

def solve_det_mod(M, b=None):
    """Gaussian elimination mod PR: returns (det, solution or None)."""
    n = len(M); A = [row[:] + ([b[i]] if b is not None else []) for i, row in enumerate(M)]; det = 1
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] % PR), None)
        if piv is None: return 0, None
        if piv != c: A[c], A[piv] = A[piv], A[c]; det = -det
        det = det*A[c][c] % PR; inv = pow(A[c][c], PR - 2, PR)
        A[c] = [x*inv % PR for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]; A[r] = [(x - f*y) % PR for x, y in zip(A[r], A[c])]
    return det % PR, ([A[i][n] for i in range(n)] if b is not None else None)

def kahan_mod(fld, z, h):
    n = fld.n; Jz = fld.J_at(z); half = h*pow(2, PR - 2, PR) % PR
    M = [[((1 if i == j else 0) - half*Jz[i][j]) % PR for j in range(n)] for i in range(n)]
    dM, sol = solve_det_mod(M, fld.F_at(z))
    if dM == 0: return None
    zn = [(zi + h*si) % PR for zi, si in zip(z, sol)]
    Jn = fld.J_at(zn)
    Mp = [[((1 if i == j else 0) + half*Jn[i][j]) % PR for j in range(n)] for i in range(n)]
    dMp, _ = solve_det_mod(Mp)
    return zn, dMp*pow(dM, PR - 2, PR) % PR

def monomials(n, d):
    out = [()]
    for k in range(1, d + 1): out += list(combinations_with_replacement(range(n), k))
    return out

def eval_monos(monos, z):
    vals = {(): 1}; res = []
    for m in monos:
        if m not in vals: vals[m] = vals[m[:-1]]*z[m[-1]] % PR
        res.append(vals[m])
    return res

def nullity_mod(A):
    A = np.array(A, dtype=np.int64) % PR; rows, cols = A.shape; r = 0
    for c in range(cols):
        nz = np.nonzero(A[r:, c])[0]
        if len(nz) == 0: continue
        p = r + nz[0]
        if p != r: A[[r, p]] = A[[p, r]]
        A[r] = A[r]*pow(int(A[r, c]), PR - 2, PR) % PR
        col = A[:, c].copy(); col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr):
            A[nzr] = (A[nzr] - (col[nzr, None]*A[r][None, :]) % PR) % PR
        r += 1
        if r == rows: break
    return cols - r

def density_nullity(F, vars, d, extra_rows=40, seed=1):
    fld = ModField(F, vars); rng = random.Random(seed); h = rng.randrange(1, PR)
    monos = monomials(len(vars), d); rows = []
    while len(rows) < len(monos) + extra_rows:
        z = [rng.randrange(PR) for _ in vars]
        out = kahan_mod(fld, z, h)
        if out is None: continue
        zn, det = out
        a, b = eval_monos(monos, zn), eval_monos(monos, z)
        rows.append([(x - det*y) % PR for x, y in zip(a, b)])
    return nullity_mod(rows), len(monos)
