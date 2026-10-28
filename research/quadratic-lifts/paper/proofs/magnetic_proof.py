"""Planar isotropic oscillator in a uniform magnetic field, lifted with y = |q|^2:
    q' = v,   v' = -2 U'(y) q + B J v,   y' = 2 q.v,     U(y) = u2 y^2/2,   J v = (-v2, v1)
(u1 = 0 without loss of generality, by the shift y -> y + u1/u2).  Exact verification, with h, B, u2 symbolic,
of the statements of the proposition on the magnetic case:
  0. the closed form of Kahan's map (complex notation Q = q1 + i q2, V = v1 + i v2, beta = h B/2)
        Q' = ((2 - Y' - i beta) Q + h V)/(Y - i beta),  V' = (-(al + al') Q + (2 - Y + i beta) V)/(Y - i beta),
     with al = h U'(y), Y = 1 + h al/2 and y' the root of a scalar linear equation, satisfies Kahan's equations;
  1. the discrete Larmor identity  Y q^q' - beta q.q' = h (q^v - B/2 |q|^2);
  2. conservation of c_h, L_h = P_L/P^B and Ecal_h = Phat_E/((2 + kap c_h) P^B);
  3. det(I - h/2 F'(z)) = Y (Y + kap q.eta) + beta^2 + h beta kap M/2,  eta = q + h v/2;
  4. the measure identity  P^B(z') det(I - h/2 F'(z)) = det(I + h/2 F'(z')) P^B(z);
  5. the reductions at B = 0 and the limits h -> 0;
  6. functional independence of (c_h, L_h, Ecal_h).
The map and all functions are invariant under rotations of the plane (the field commutes with J), so the
identities are checked on the slice q = (q1, 0); the image point is not restricted.  The next state is written
as (polynomial)/Den with Den = Y^2 + beta^2, and y' is eliminated at the end through its linear equation
a y' + b = 0 (a != 0).  Sparse polynomial arithmetic over QQ; total run time a few seconds."""
import sympy as sp, time
from sympy import QQ
from sympy.polys.rings import ring
t00 = time.time()
R, q1, v1, v2, y, yn, h, B, u2 = ring("q1 v1 v2 y yn h B u2", QQ)
kap, beta = h**2*u2, h*B/2
Y, Yn = 1 + kap*y/2, 1 + kap*yn/2
al, aln = h*u2*y, h*u2*yn
Den = Y**2 + beta**2
cmul = lambda a, b: (a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0])
cadd = lambda a, b: (a[0] + b[0], a[1] + b[1])
cscale = lambda c, a: (c*a[0], c*a[1])
conj = lambda a: (a[0], -a[1])
Qc, Vc, Wbar = (q1, R(0)), (v1, v2), (Y, beta)        # Wbar = conj(Y - i beta)
QnD = cmul(cadd(cmul((2 - Yn, -beta), Qc), cscale(h, Vc)), Wbar)                   # Den * Q'
VnD = cmul(cadd(cscale(-(al + aln), Qc), cmul((2 - Y, beta), Vc)), Wbar)           # Den * V'
lin = (yn - y)*Den - h*(cmul(conj(Qc), VnD)[0] + cmul(conj(QnD), Vc)[0])            # y' - y = h Re(conj(Q)V' + conj(Q')V)
assert lin.degree(yn) == 1
a_, b_ = lin.coeff_wrt(yn, 1), lin.coeff_wrt(yn, 0)
def at_yn(E):
    "a^deg E(y' = -b/a): zero iff E vanishes on the graph of the map."
    if E == 0:
        return R(0)
    d = E.degree(yn)
    return sum((E.coeff_wrt(yn, k)*(-b_)**k*a_**(d - k) for k in range(d + 1)), R(0))
def report(label, ok):
    print(f"{label}: {ok}  ({time.time() - t00:.1f}s)", flush=True)
    assert ok
# 0. Kahan's equations, multiplied by Den
qn, vn = QnD, VnD
Jsum = (-(v2*Den + vn[1]), v1*Den + vn[0])                                          # Den * J(v + v')
eqs = [qn[0] - q1*Den - h/2*(v1*Den + vn[0]), qn[1] - h/2*(v2*Den + vn[1]),
       vn[0] - v1*Den - h*(-u2*(y*qn[0] + yn*q1*Den) + B/2*Jsum[0]),
       vn[1] - v2*Den - h*(-u2*y*qn[1] + B/2*Jsum[1]),
       (yn - y)*Den - h*(q1*vn[0] + qn[0]*v1 + qn[1]*v2)]
report("[0] closed form satisfies Kahan's equations", all(at_yn(e) == 0 for e in eqs))
# 1. discrete Larmor identity: Y q^q' - beta q.q' = h (q^v - B/2 |q|^2)   (times Den)
wedge = q1*qn[1]; dot = q1*qn[0]
report("[1] discrete Larmor identity", at_yn(Y*wedge - beta*dot - h*(q1*v2 - B/2*q1**2)*Den) == 0)
# 2. integrals.  For the next state q = qn/Den, v = vn/Den; every polynomial is homogenised with s = Den.
def polys(qx, qy, vx, vy, yy, s):
    """Returns s^2 c_h, s^2 P^B, s^4 P_L, s^4 Phat_E for the state (q, v) = (qx, qy, vx, vy)/s."""
    s2 = s*s
    a = qx*qx + qy*qy; c = vx*vx + vy*vy; M = qx*vy - qy*vx                         # s^2 |q|^2, s^2 |v|^2, s^2 q^v
    Yv = 1 + kap*yy/2
    ch = yy*s2 - a + h**2/4*c                                                       # s^2 c_h
    D = (2 - Yv)*Yv*s2 + kap*(yy*s2 - ch)                                           # s^2 D
    K = c/2 + u2*yy**2/2*s2                                                         # s^2 K
    PB = Yv*(D - beta**2*s2) + 2*beta**2*s2 + h*beta*kap*M
    PL = M*(D + beta**2*s2) - B*(2*a*Yv*s2 + (1 + Yv)*ch*s2 + yy/2*(Yv**2 - 2*Yv - 2 + beta**2)*s2*s2)
    PE = ((2*s2 + kap*ch)*K*Yv + kap/2*M**2*(4 - Yv)
          + B**2/2*(yy*s2*s2 - (a + ch)*(2 - Yv)*s2) - kap*B*M*(yy/2*s2 + a + ch))
    return ch, PB, PL, PE
one = R(1)
ch0, PB0, PL0, PE0 = polys(q1, R(0), v1, v2, y, one)
ch1, PB1, PL1, PE1 = polys(qn[0], qn[1], vn[0], vn[1], yn, Den)
D2 = Den**2
report("[2] c_h conserved", at_yn(ch1 - ch0*D2) == 0)
# L_h = P_L/P^B: (PL1/Den^4)/(PB1/Den^2) = PL0/PB0
report("[2] L_h = P_L/P^B conserved", at_yn(PL1*PB0 - PL0*PB1*D2) == 0)
# Ecal_h = Phat_E/((2 + kap c_h) P^B), and c_h is conserved: (PE1/Den^4)/(PB1/Den^2) = PE0/PB0
report("[2] Ecal_h = Phat_E/((2 + kap c_h) P^B) conserved", at_yn(PE1*PB0 - PE0*PB1*D2) == 0)
# 3. the two determinants, in closed form
Q1, Q2, V1, V2, Ys, hs, Bs, u2s = sp.symbols('Q1 Q2 V1 V2 Ys h B u2')
Fs = sp.Matrix([V1, V2, -2*u2s*Ys*Q1 - Bs*V2, -2*u2s*Ys*Q2 + Bs*V1, 2*(Q1*V1 + Q2*V2)])
Js = Fs.jacobian([Q1, Q2, V1, V2, Ys])
kaps, betas, Yss = hs**2*u2s, hs*Bs/2, 1 + hs**2*u2s*Ys/2
Ms = Q1*V2 - Q2*V1
for sgn, name in [(-1, "det(I - h/2 F')"), (1, "det(I + h/2 F')")]:
    det = (sp.eye(5) + sgn*hs/2*Js).det()
    etas = (Q1 - sgn*hs/2*V1, Q2 - sgn*hs/2*V2)
    closed = Yss*(Yss + kaps*(Q1*etas[0] + Q2*etas[1])) + betas**2 + hs*betas*kaps*Ms/2
    report(f"[3] {name} = Y (Y + kap q.(q {'+' if sgn < 0 else '-'} h v/2)) + beta^2 + h beta kap M/2",
           sp.expand(det - closed) == 0)
# 4. measure identity, with the closed forms of the determinants
def detpoly(qx, qy, vx, vy, yy, s, sgn):
    "s^2 det(I + sgn h/2 F') at the state (qx, qy, vx, vy)/s."
    Yv = 1 + kap*yy/2
    qeta = qx*(qx - sgn*h/2*vx) + qy*(qy - sgn*h/2*vy)
    return Yv*(Yv*s*s + kap*qeta) + beta**2*s*s + h*beta*kap*(qx*vy - qy*vx)/2
dm = detpoly(q1, R(0), v1, v2, y, one, -1)
dp = detpoly(qn[0], qn[1], vn[0], vn[1], yn, Den, 1)
report("[4] measure identity P^B(z') det(I - h/2 F'(z)) = det(I + h/2 F'(z')) P^B(z)", at_yn(PB1*dm - dp*PB0) == 0)
# 5. reductions and limits (expression level)
q1s, v1s, v2s, ys = sp.symbols('q1 v1 v2 y')
sub = lambda e: e.as_expr().subs({sp.Symbol('B'): 0})
Yx = 1 + sp.Symbol('h')**2*sp.Symbol('u2')*sp.Symbol('y')/2
hh, uu = sp.Symbol('h'), sp.Symbol('u2')
kx = hh**2*uu
ax, cx, Mx = q1s**2, v1s**2 + v2s**2, q1s*v2s
chx = ys - ax + hh**2/4*cx
Dx = (2 - Yx)*Yx + kx*(ys - chx)
Kx = cx/2 + uu*ys**2/2
Eh = (Kx + kx/2*(Mx/Yx)**2)/Dx
report("[5] B = 0: P^B = Y D", sp.simplify(sub(PB0) - Yx*Dx) == 0)
report("[5] B = 0: L_h = M/Y = M_h", sp.simplify(sub(PL0)/sub(PB0) - Mx/Yx) == 0)
report("[5] B = 0: Ecal_h = E_h + kap M_h^2/(2(2 + kap c_h))",
       sp.simplify(sub(PE0)/((2 + kx*chx)*sub(PB0)) - Eh - kx*(Mx/Yx)**2/(2*(2 + kx*chx))) == 0)
Bx = sp.Symbol('B')
Lh = PL0.as_expr()/PB0.as_expr()
Ecal = PE0.as_expr()/((2 + kx*chx)*PB0.as_expr())
report("[5] h -> 0: L_h -> q^v - B/2 y", sp.simplify(sp.limit(Lh, hh, 0) - (Mx - Bx/2*ys)) == 0)
report("[5] h -> 0: Ecal_h -> |v|^2/2 + u2 y^2/2", sp.simplify(sp.limit(Ecal, hh, 0) - Kx) == 0)
# 6. functional independence: a 3x3 minor of the Jacobian of (c_h, L_h, Ecal_h) with respect to (q1, v2, y),
#    evaluated exactly at a rational point (non-zero there, hence not identically zero)
F3 = sp.Matrix([chx, Lh, Ecal])
Jm = F3.jacobian([q1s, v2s, ys])
pt = {q1s: 1, v1s: sp.Rational(1, 3), v2s: sp.Rational(1, 2), ys: sp.Rational(2, 3), hh: sp.Rational(1, 5),
      Bx: sp.Rational(1, 2), uu: 1}
minor = sp.nsimplify(Jm.subs(pt).det())
report(f"[6] Jacobian minor of (c_h, L_h, Ecal_h) at a rational point = {minor} != 0", minor != 0)
