"""How the density of Proposition 5.3 was found.  Search for a density dz/P of the lifted Kahan map of the general
quartic 1-DOF Nambu lift
   V = a4 q^4 + a3 q^3 + a2 q^2 + a1 q,  Vt = a4 y^2 + a3 q y + a2 y + a1 q,
   q' = p, p' = -4 a4 q y - 2 a3 q^2 - 2 a2 q - a3 y - a1, y' = 2 q p,
with P a polynomial in (t = h^2, q, p, y, a4, a3, a2, a1) that is homogeneous of weight 0 for the two gradings
   A: q 1, p 2, y 2, t -2, a4 0, a3 1, a2 2, a1 3;   B: q 0, p 1, y 0, t -2, a4 2, a3 2, a2 2, a1 2
and of degree <= DZ in (q, p, y) and <= KMAX in t.  The coefficients are rational numbers; the parameters and h are
random residues, so a solution is valid for generic parameters.  Solved modulo four primes, combined by the
Chinese remainder theorem, and recovered by rational reconstruction.  With DZ = 4, KMAX = 4 the null space is
spanned by P and h^2 a2 P (a factor that depends only on the parameters), where P is the density of
Proposition 5.3; with DZ <= 3 it is trivial.  Usage: python3 quartic_density_search.py [DZ KMAX]; a few seconds."""
import random, sys
from fractions import Fraction as Fr
from math import isqrt
from sympy import prevprime
DZ = int(sys.argv[1]) if len(sys.argv) > 1 else 4
KMAX = int(sys.argv[2]) if len(sys.argv) > 2 else 4
mons = []
for k in range(KMAX + 1):
    for e1 in range(DZ + 1):
        for e2 in range(0, DZ + 1 - e1):
            for e3 in range(DZ + 1 - e1 - e2):
                for f4 in range(k + 1):
                    for f3 in range(k + 1 - f4):
                        for f2 in range(k + 1 - f4 - f3):
                            for f1 in range(k + 1 - f4 - f3 - f2):
                                A = -2*k + e1 + 2*e2 + 2*e3 + f3 + 2*f2 + 3*f1
                                Bg = -2*k + e2 + 2*(f4 + f3 + f2 + f1)
                                if A == 0 and Bg == 0:
                                    mons.append((k, e1, e2, e3, f4, f3, f2, f1))
print(len(mons), "unknowns", flush=True)
def field(z, a, P):
    q, p, y = z; a4, a3, a2, a1 = a
    F = [p, (-4*a4*q*y - 2*a3*q*q - 2*a2*q - a3*y - a1) % P, 2*q*p % P]
    J = [[0, 1, 0], [(-4*a4*y - 4*a3*q - 2*a2) % P, 0, (-4*a4*q - a3) % P], [2*p % P, 2*q % P, 0]]
    return F, J
def solve3(M, b, P):
    n = 3; A = [row[:] + [b[i]] for i, row in enumerate(M)]; det = 1
    for c in range(n):
        piv = next((r for r in range(c, n) if A[r][c] % P), None)
        if piv is None: return None, 0
        if piv != c: A[c], A[piv] = A[piv], A[c]; det = -det
        det = det*A[c][c] % P; inv = pow(A[c][c], P - 2, P); A[c] = [x*inv % P for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]; A[r] = [(x - f*yy) % P for x, yy in zip(A[r], A[c])]
    return [A[i][n] for i in range(n)], det % P
def ev(m, t, z, a, P):
    k, e1, e2, e3, f4, f3, f2, f1 = m; q, p, y = z; a4, a3, a2, a1 = a
    return pow(t, k, P)*pow(q, e1, P)*pow(p, e2, P)*pow(y, e3, P)*pow(a4, f4, P)*pow(a3, f3, P)*pow(a2, f2, P)*pow(a1, f1, P) % P
def nullspace(rows, n, P):
    A = [r[:] for r in rows]; piv = []; r = 0
    for c in range(n):
        pr = next((i for i in range(r, len(A)) if A[i][c] % P), None)
        if pr is None: continue
        A[r], A[pr] = A[pr], A[r]; inv = pow(A[r][c], P - 2, P); A[r] = [x*inv % P for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]; A[i] = [(x - f*yy) % P for x, yy in zip(A[i], A[r])]
        piv.append(c); r += 1
        if r == len(A): break
    free = [c for c in range(n) if c not in piv]; basis = []
    for fc in free:
        v = [0]*n; v[fc] = 1
        for i, pc in enumerate(piv): v[pc] = (-A[i][fc]) % P
        basis.append(v)
    return free, basis
def ratrec(x, N):
    r0, r1, s0, s1 = N, x % N, 0, 1; bound = isqrt(N//2)
    while r1 > bound: qq = r0//r1; r0, r1 = r1, r0 - qq*r1; s0, s1 = s1, s0 - qq*s1
    if s1 == 0 or abs(s1) > bound: return None
    return Fr(r1, s1) if s1 > 0 else Fr(-r1, -s1)
n = len(mons); rnd = random.Random(5); res = []; P = 2**31 - 1
for it in range(4):
    P = prevprime(P); rows = []
    while len(rows) < n + 40:
        h = rnd.randrange(1, P); t = h*h % P; half = h*pow(2, P - 2, P) % P
        z = [rnd.randrange(P) for _ in range(3)]; a = [rnd.randrange(P) for _ in range(4)]
        F, J = field(z, a, P)
        M = [[((1 if i == j else 0) - half*J[i][j]) % P for j in range(3)] for i in range(3)]
        sol, dm = solve3(M, F, P)
        if sol is None: continue
        zn = [(zi + h*si) % P for zi, si in zip(z, sol)]
        _, Jn = field(zn, a, P)
        Mp = [[((1 if i == j else 0) + half*Jn[i][j]) % P for j in range(3)] for i in range(3)]
        _, dp = solve3(Mp, [0, 0, 0], P)
        dt = dp*pow(dm, P - 2, P) % P
        rows.append([(ev(m, t, zn, a, P) - dt*ev(m, t, z, a, P)) % P for m in mons])
    free, basis = nullspace(rows, n, P)
    print(f"prime {P}: nullity {len(basis)}", flush=True)
    res.append((P, free, basis))
frees = [r[1] for r in res]
assert all(f == frees[0] for f in frees), "inconsistent pivots; try other primes"
if not res[0][2]:
    print("no density in this ansatz"); raise SystemExit
N = 1; comb = None
for P, free, basis in res:
    if comb is None: comb, N = [v[:] for v in basis], P; continue
    comb = [[x + N*(((y - x) % P)*pow(N % P, P - 2, P) % P) for x, y in zip(vo, vn)] for vo, vn in zip(comb, basis)]; N *= P
vecs = [[ratrec(x, N) for x in v] for v in comb]
assert all(c is not None for v in vecs for c in v), "rational reconstruction failed; use more primes"
import sympy as sp
t, q, p, y, a4, a3, a2, a1 = sp.symbols('t q p y a4 a3 a2 a1')
for v in vecs:
    e = sum(sp.Rational(c.numerator, c.denominator)*t**m[0]*q**m[1]*p**m[2]*y**m[3]*a4**m[4]*a3**m[5]*a2**m[6]*a1**m[7]
            for m, c in zip(mons, v) if c)
    print("density (t = h^2):", sp.factor(e))
