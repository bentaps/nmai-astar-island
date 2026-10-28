"""Proposition 5.3: for every potential V = a4 q^4 + a3 q^3 + a2 q^2 + a1 q, Kahan's map of the Nambu lift
    q' = p,   p' = -Vt_q - 2 q Vt_y = -(4 a4 q y + 2 a3 q^2 + 2 a2 q + a3 y + a1),   y' = 2 q p,
    Vt = a4 y^2 + a3 q y + a2 y + a1 q,
preserves dq dp dy / P with
    P  = 3 + h^2 T1 + h^4 T2/16 + h^6 (4 a4 q + a3)^2 p^2/16,
    T1 = 4 a4 q^2 + 2 a4 y + 3 a3 q + a2,
    T2 = -4 (2 a4 w + a3 q + a2)^2 + 8 (4 a2 a4 - a3^2) w + 8 a1 (4 a4 q + a3),   w = 2 q^2 - y.
With M = I - h/2 F'(z), Delta = det M, the step is z' = z + h adj(M) F(z)/Delta.  Multiplying the measure identity
    P(z') det(I - h/2 F'(z)) = det(I + h/2 F'(z')) P(z)
by Delta^4 gives an identity between polynomials in (q, p, y, h, a4, a3, a2, a1), checked here exactly with sparse
polynomial arithmetic over QQ (a fraction of a second).  Also checks the values of T1, 4 a4 q + a3 and T2 on the
sheet y = q^2 and the special case V = q^4/4 + q^3/3 + q^2/2 of code-invariants/measure_nonsym_allh.py."""
import time
import sympy as sp
from sympy import QQ
from sympy.polys.rings import ring
t0 = time.time()
R, q, p, y, h, a4, a3, a2, a1 = ring("q p y h a4 a3 a2 a1", QQ)
t = h**2
def F_hom(q_, p_, y_, s):
    "s^2 F(z/s)"
    return [p_*s, -(4*a4*q_*y_ + 2*a3*q_*q_ + 2*a2*q_*s + a3*y_*s + a1*s*s), 2*q_*p_]
def J_hom(q_, p_, y_, s):
    "s F'(z/s)"
    return [[R(0), s, R(0)], [-(4*a4*y_ + 4*a3*q_ + 2*a2*s), R(0), -(4*a4*q_ + a3*s)], [2*p_, 2*q_, R(0)]]
def det3(M):
    return (M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
            + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]))
def adj3(M):
    minor = lambda i, j: [[M[r][c] for c in range(3) if c != j] for r in range(3) if r != i]
    d2 = lambda A: A[0][0]*A[1][1] - A[0][1]*A[1][0]
    return [[(-1)**(i + j)*d2(minor(j, i)) for j in range(3)] for i in range(3)]
def P_hom(q_, p_, y_, s):
    "s^4 P(z/s)"
    s2 = s*s
    T1 = 4*a4*q_*q_ + 2*a4*y_*s + 3*a3*q_*s + a2*s2
    w = 2*q_*q_ - y_*s
    T2 = -4*(2*a4*w + a3*q_*s + a2*s2)**2 + 8*w*(4*a2*a4 - a3**2)*s2 + 8*a1*(4*a4*q_ + a3*s)*s2*s
    return 3*s2*s2 + t*T1*s2 + t**2/16*T2 + t**3/16*(4*a4*q_ + a3*s)**2*p_**2
one = R(1)
J = J_hom(q, p, y, one)
M = [[(1 if i == j else 0) - h/2*J[i][j] for j in range(3)] for i in range(3)]
Delta, A = det3(M), adj3(M)
assert all(sum(M[i][k]*A[k][j] for k in range(3)) == (Delta if i == j else 0) for i in range(3) for j in range(3))
F = F_hom(q, p, y, one)
Z = [zi*Delta + h*sum(A[i][k]*F[k] for k in range(3)) for i, zi in enumerate((q, p, y))]   # Delta z'
Jn = J_hom(Z[0], Z[1], Z[2], Delta)                                                      # Delta F'(z')
Mp = [[(Delta if i == j else 0) + h/2*Jn[i][j] for j in range(3)] for i in range(3)]    # Delta (I + h/2 F'(z'))
# P(z') Delta = det(I + h/2 F'(z')) P(z)   <=>   Delta^4 P(z') = Delta^3 det(I + h/2 F'(z')) P(z)
ok = P_hom(Z[0], Z[1], Z[2], Delta) == det3(Mp)*P_hom(q, p, y, one)
print(f"[1] measure identity for all (q, p, y, h, a4, a3, a2, a1): {ok}  ({time.time() - t0:.1f}s)")
assert ok
# values on the sheet y = q^2
qs, A4, A3, A2, A1 = sp.symbols('q a4 a3 a2 a1')
V = A4*qs**4 + A3*qs**3 + A2*qs**2 + A1*qs
sheet = lambda e: e.as_expr().subs({sp.Symbol('y'): qs**2})
T1 = 4*a4*q**2 + 2*a4*y + 3*a3*q + a2
w = 2*q**2 - y
T2 = -4*(2*a4*w + a3*q + a2)**2 + 8*w*(4*a2*a4 - a3**2) + 8*a1*(4*a4*q + a3)
d = lambda k: sp.diff(V, qs, k)
checks = [sp.expand(sheet(T1) - d(2)/2) == 0, sp.expand(sheet(4*a4*q + a3) - d(3)/6) == 0,
          sp.expand(sheet(T2) - (sp.Rational(4, 3)*d(1)*d(3) - d(2)**2)) == 0]
print(f"[2] on y = q^2: T1 = V''/2, 4 a4 q + a3 = V'''/6, T2 = 4/3 V' V''' - V''^2: {all(checks)}")
assert all(checks)
# special case of code-invariants/measure_nonsym_allh.py (normalised there by 432 = 144 * 3)
P = P_hom(q, p, y, one).as_expr()
hs, ps, ys = sp.symbols('h p y')
Pspec = P.subs({A4: sp.Rational(1, 4), A3: sp.Rational(1, 3), A2: sp.Rational(1, 2), A1: 0})
tt = hs**2
Pold = (432 + 72*tt*(1 + 2*qs + 2*qs**2 + ys) + tt**2*(-9 - 12*qs - 10*ys + 16*qs**2 + 12*qs*ys - 9*ys**2 - 24*qs**3
        + 36*qs**2*ys - 36*qs**4) + tt**3*ps**2*(1 + 3*qs)**2)
print(f"[3] V = q^4/4 + q^3/3 + q^2/2 reproduces the earlier closed form (times 144): {sp.expand(144*Pspec - Pold) == 0}")
# factorisation for V = q^4/4:  P = (1 + h^2 q^2/2)(3 + h^2 c_h/2 - h^4 Ht_h/4),  Ht_h = (p^2/2 + y^2/4)/(1 + h^2 q^2/2)
P4 = P.subs({A4: sp.Rational(1, 4), A3: 0, A2: 0, A1: 0})
den = 1 + hs**2*qs**2/2
ch = ys - qs**2 + hs**2/4*ps**2
Hh = (ps**2/2 + ys**2/4)/den
print(f"[4] V = q^4/4: P = (1 + h^2 q^2/2)(3 + h^2 c_h/2 - h^4 Ht_h/4): {sp.simplify(P4 - den*(3 + hs**2*ch/2 - hs**4*Hh/4)) == 0}")
# the h^2 and h^6 terms as invariants of the Jacobian: h^2 T1 = h^2 sigma_2(F')/2, h^6 (4a4 q + a3)^2 p^2/16 = h^6 det(F')^2/64
Fq = sp.Matrix([ps, -(A4*4*qs*ys + 2*A3*qs**2 + 2*A2*qs + A3*ys + A1), 2*qs*ps]); Jq = Fq.jacobian([qs, ps, ys])
sig2 = sum(Jq.extract([i, j], [i, j]).det() for i in range(3) for j in range(i + 1, 3))
T1e = 4*A4*qs**2 + 2*A4*ys + 3*A3*qs + A2
print(f"[5] sigma_2(F') = 2 T1 and det(F')^2/64 = (4a4 q + a3)^2 p^2/16: "
      f"{sp.expand(sig2 - 2*T1e) == 0 and sp.expand(Jq.det()**2/64 - (4*A4*qs + A3)**2*ps**2/16) == 0}")
