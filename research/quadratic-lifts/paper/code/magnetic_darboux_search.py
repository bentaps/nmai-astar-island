"""Search for the invariant densities dz/P of the lifted Kahan map of the planar magnetic isotropic oscillator
(Proposition 7.2), q' = v, v' = -2 U'(y) q + B J v, y' = 2 q.v, U(y) = u2 y^2/2 + u1 y, for fixed rational
parameters.  P is sought in the basis of rotation-invariant monomials y^i a^j b^k c^l M^m (a = |q|^2, b = q.v,
c = |v|^2, M = q1 v2 - q2 v1) of weighted degree <= d (y has weight 1, a, b, c weight 2, M <= 1 since M^2 = ac - b^2).
The density condition P(z') = det DPhi(z) P(z) is imposed at random integer points modulo several primes; the
null spaces are combined by the Chinese remainder theorem and the coefficients recovered by rational
reconstruction.  Ratios of two densities are rational integrals; this is how P^B, P_L and Phat_E were found.
Usage: python3 magnetic_darboux_search.py h B u2 u1    (rationals, e.g. 1/5 1 1/2 0); about a second."""
import sys, random
from fractions import Fraction as Fr
from sympy import prevprime
from math import isqrt
def basis_list(d):
    out = []
    for i in range(d + 1):
        for w in range((d - i)//2 + 1):
            for mm in (0, 1):
                if mm > w: continue
                for j in range(w - mm + 1):
                    for k in range(w - mm - j + 1):
                        out.append((i, j, k, w - mm - j - k, mm))
    return out
def field(z, P):                       # lifted magnetic field and Jacobian (mod P), parameters in globals
    q1, q2, v1, v2, y = z; U = (u2*y + u1) % P
    F = [v1, v2, (-2*U*q1 - B*v2) % P, (-2*U*q2 + B*v1) % P, 2*(q1*v1 + q2*v2) % P]
    J = [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0],
         [(-2*U) % P, 0, 0, (-B) % P, (-2*u2*q1) % P],
         [0, (-2*U) % P, B % P, 0, (-2*u2*q2) % P],
         [2*v1 % P, 2*v2 % P, 2*q1 % P, 2*q2 % P, 0]]
    return F, J
def solve(M, b, P):
    n = len(M); A = [row[:] + [b[i]] for i, row in enumerate(M)]; det = 1
    for c in range(n):
        p = next((r for r in range(c, n) if A[r][c] % P), None)
        if p is None: return None, 0
        if p != c: A[c], A[p] = A[p], A[c]; det = -det
        det = det*A[c][c] % P; inv = pow(A[c][c], P - 2, P); A[c] = [x*inv % P for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]; A[r] = [(x - f*y) % P for x, y in zip(A[r], A[c])]
    return [A[i][n] for i in range(n)], det % P
def detm(M, P):
    _, d = solve(M, [0]*len(M), P); return d
def ev_basis(bs, z, P):
    q1, q2, v1, v2, y = z
    a = (q1*q1 + q2*q2) % P; b = (q1*v1 + q2*v2) % P; c = (v1*v1 + v2*v2) % P; M = (q1*v2 - q2*v1) % P
    return [pow(y, i, P)*pow(a, j, P)*pow(b, k, P)*pow(c, l, P)*pow(M, m, P) % P for (i, j, k, l, m) in bs]
def nullspace_rref(rows, n, P):
    A = [r[:] for r in rows]; piv = []; r = 0
    for c in range(n):
        p = next((i for i in range(r, len(A)) if A[i][c] % P), None)
        if p is None: continue
        A[r], A[p] = A[p], A[r]; inv = pow(A[r][c], P - 2, P); A[r] = [x*inv % P for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]; A[i] = [(x - f*y) % P for x, y in zip(A[i], A[r])]
        piv.append(c); r += 1
        if r == len(A): break
    free = [c for c in range(n) if c not in piv]; basis = []
    for fc in free:
        v = [0]*n; v[fc] = 1
        for i, pc in enumerate(piv): v[pc] = (-A[i][fc]) % P
        basis.append(v)
    return free, basis
def ratrec(x, N):
    # rational reconstruction of x mod N
    r0, r1, s0, s1 = N, x % N, 0, 1; bound = isqrt(N//2)
    while r1 > bound: qq = r0//r1; r0, r1 = r1, r0 - qq*r1; s0, s1 = s1, s0 - qq*s1
    if s1 == 0 or abs(s1) > bound: return None
    return Fr(r1, s1) if s1 > 0 else Fr(-r1, -s1)
def run(hq, Bq, u2q, u1q, d=5, nprimes=6, seed=3):
    global u2, u1, B
    bs = basis_list(d); n = len(bs); rnd = random.Random(seed)
    pts = [[rnd.randint(-50, 50) for _ in range(5)] for _ in range(n + 30)]
    res = []; P = 2**31 - 1; primes = []
    for _ in range(nprimes):
        P = prevprime(P); primes.append(P)
        tomod = lambda f: f.numerator % P*pow(f.denominator % P, P - 2, P) % P
        h, B, u2, u1 = tomod(hq), tomod(Bq), tomod(u2q), tomod(u1q); half = h*pow(2, P - 2, P) % P
        rows = []
        for z in pts:
            z = [x % P for x in z]; F, J = field(z, P)
            Mm = [[((1 if i == j else 0) - half*J[i][j]) % P for j in range(5)] for i in range(5)]
            sol, dm = solve(Mm, F, P)
            if sol is None: continue
            zn = [(zi + h*si) % P for zi, si in zip(z, sol)]; _, Jn = field(zn, P)
            Mp = [[((1 if i == j else 0) + half*Jn[i][j]) % P for j in range(5)] for i in range(5)]
            dt = detm(Mp, P)*pow(dm, P - 2, P) % P
            A1, A0 = ev_basis(bs, zn, P), ev_basis(bs, z, P)
            rows.append([(x - dt*y) % P for x, y in zip(A1, A0)])
        free, basis = nullspace_rref(rows, n, P)
        res.append((P, free, basis))
    frees = [r[1] for r in res]; assert all(f == frees[0] for f in frees), "inconsistent free columns"
    N = 1; combined = None
    for P, free, basis in res:
        if combined is None: combined = [[x for x in v] for v in basis]; N = P; continue
        newc = []
        for v_old, v_new in zip(combined, basis):
            vv = []
            for x, y in zip(v_old, v_new):
                t = ((y - x) % P)*pow(N % P, P - 2, P) % P; vv.append(x + N*t)
            newc.append(vv)
        combined = newc; N *= P
    dens = []
    for v in combined:
        cf = [ratrec(x, N) for x in v]
        if any(c is None for c in cf): raise RuntimeError("reconstruction failed; use more primes")
        dens.append(cf)
    return bs, frees[0], dens
if __name__ == "__main__":
    hq, Bq, u2q, u1q = [Fr(x) for x in sys.argv[1:5]]
    bs, free, dens = run(hq, Bq, u2q, u1q)
    names = [f"y^{i} a^{j} b^{k} c^{l} M^{m}" for (i, j, k, l, m) in bs]
    print(f"{len(dens)} invariant densities; free columns {[names[f] for f in free]}")
    for v in dens:
        print({names[i]: str(c) for i, c in enumerate(v) if c != 0})
