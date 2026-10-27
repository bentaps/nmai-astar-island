"""Computer-assisted proof of Theorem 6.1(c), and an independent check of Theorem 6.1(b), for the lifted isotropic
oscillator H = |p|^2/2 + U(|q|^2), U(y) = u2 y^2/2 + u1 y, lift y = |q|^2, with h, u1, u2 symbolic. All checks are
exact rational-function identities. (Theorem 6.1(b) also has a proof by hand in the paper.)

0. The closed-form step of Remark 6.4 satisfies Kahan's equations identically (on all of R^{2n+1}), so it IS
   Kahan's map.
1. n = 2: c_h, M_h, E_h are conserved. Both the map and the invariants commute with rotations, and every point can
   be rotated to q2 = 0, so the identities are checked on that slice (the image point is not restricted).
2. n = 2: the measure identity P_2(z') det(I - h/2 F'(z)) = det(I + h/2 F'(z')) P_2(z).
3. n = 3: the same identities for E_h (with |M_h|^2) and P_3 = Y^2 K-factor, on the slice q = (q1,0,0),
   p = (p1,p2,0), which every point can be rotated to; this checks the reduction argument for n > 2 directly.
4. The other form of E_h used in the notes satisfies 1/E_h(alt) = 1/E_h + h^4 u2/2, and c_h, M_h, E_h are
   functionally independent (a 3x3 Jacobian minor is non-zero).
Usage: python3 isotropic_proof.py"""
import sympy as sp, time
h, u2, u1 = sp.symbols('h u2 u1')
Up = lambda Y: u2*Y + u1

def setup(n):
    q = sp.symbols(f'q1:{n+1}'); p = sp.symbols(f'p1:{n+1}'); y = sp.Symbol('y')
    Z = sp.Matrix([*q, *p, y])
    F = sp.Matrix([*p, *[-2*Up(y)*qi for qi in q], 2*sum(a*b for a, b in zip(q, p))])
    return q, p, y, Z, F

def step(q, p, y):                       # closed form of Remark 6.4
    al = h*Up(y); Yf = 1 + h*al/2
    a = sum(x*x for x in q); b = sum(x*w for x, w in zip(q, p)); c = sum(w*w for w in p)
    yn = (y*Yf - h*(al + h*u1)*(a + h*b/2) + 2*h*b + h**2*c)/(Yf + h**2*u2*(a + h*b/2))
    aln = h*Up(yn)
    qn = [((1 - h*aln/2)*x + h*w)/Yf for x, w in zip(q, p)]
    pn = [(-(al + aln)*x + (1 - h*al/2)*w)/Yf for x, w in zip(q, p)]
    return qn, pn, yn

def invariants(q, p, y, n):
    Yv = 1 + h**2*Up(y)/2
    Q2 = sum(x*x for x in q); P2 = sum(w*w for w in p)
    M2 = sum((q[a]*p[b] - q[b]*p[a])**2 for a in range(n) for b in range(a + 1, n))
    K = P2/2 + u2*y**2/2 + u1*y + u1**2/(2*u2)
    ch = y - Q2 + h**2*P2/4
    Mh2 = M2/Yv**2
    Dn = 1 + h**2*u2*Q2 - h**4*u2/2*K                      # denominator of E_h = second factor of P
    Eh = (K + h**2*u2/2*Mh2)/Dn
    P = Yv**(n - 1)*Dn
    return ch, Eh, P

def zero(e): return sp.cancel(sp.together(e)) == 0

T0 = time.time()
# 0. closed form = Kahan's map, n = 2 and n = 3, full space
for n in (2, 3):
    q, p, y, Z, F = setup(n); qn, pn, yn = step(q, p, y)
    res = [qn[i] - q[i] - h/2*(p[i] + pn[i]) for i in range(n)]
    res += [pn[i] - p[i] + h*(u2*(y*qn[i] + yn*q[i]) + u1*(q[i] + qn[i])) for i in range(n)]
    res += [yn - y - h*sum(q[i]*pn[i] + qn[i]*p[i] for i in range(n))]
    print(f"[0] n={n}: closed-form step satisfies Kahan's equations identically: {all(zero(r) for r in res)}"
          f" ({time.time()-T0:.0f}s)", flush=True)

for n, slice_ in ((2, lambda q, p: {q[1]: 0}), (3, lambda q, p: {q[1]: 0, q[2]: 0, p[2]: 0})):
    q, p, y, Z, F = setup(n); sub = slice_(q, p)
    J = F.jacobian(Z); D = sp.factor((sp.eye(2*n + 1) - h/2*J).det())
    print(f"[{1 if n == 2 else 3}] n={n}: det(I - h/2 F') = {D}", flush=True)
    qs = [sp.sympify(x).subs(sub) for x in q]; ps = [sp.sympify(w).subs(sub) for w in p]
    qn, pn, yn = step(qs, ps, y)
    qn = [sp.together(e) for e in qn]; pn = [sp.together(e) for e in pn]; yn = sp.together(yn)
    ch0, Eh0, P0 = invariants(qs, ps, y, n); ch1, Eh1, P1 = invariants(qn, pn, yn, n)
    t0 = time.time(); print(f"    c_h conserved: {zero(ch1 - ch0)} ({time.time()-t0:.0f}s)", flush=True)
    if n == 2:
        Yv0 = 1 + h**2*Up(y)/2; Yv1 = 1 + h**2*Up(yn)/2
        t0 = time.time()
        print(f"    M_h conserved: {zero((qn[0]*pn[1] - qn[1]*pn[0])/Yv1 - (qs[0]*ps[1] - qs[1]*ps[0])/Yv0)}"
              f" ({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time(); print(f"    E_h conserved: {zero(Eh1 - Eh0)} ({time.time()-t0:.0f}s)", flush=True)
    Dz = D.subs(sub); Dn = D.subs(h, -h).subs(dict(zip([*q, *p, y], [*qn, *pn, yn])), simultaneous=True)
    t0 = time.time()
    print(f"    measure: P(z') det(I - h/2 F'(z)) = det(I + h/2 F'(z')) P(z): {zero(P1*Dz - Dn*P0)}"
          f" ({time.time()-t0:.0f}s)", flush=True)

# 4. functional independence of (c_h, M_h, E_h) for n = 2, and the alternative form of E_h
q, p, y, Z, F = setup(2)
Yv = 1 + h**2*Up(y)/2; M = q[0]*p[1] - q[1]*p[0]
ch, Eh, _ = invariants(q, p, y, 2); Mh = M/Yv
K = (p[0]**2 + p[1]**2)/2 + u2*y**2/2 + u1*y + u1**2/(2*u2)
Eh_alt = (K + h**2*u2/2*Mh**2)/(1 + h**2*u2*(q[0]**2 + q[1]**2) + h**6*u2**2/4*Mh**2)
print(f"[4] 1/E_h(alt) = 1/E_h + h^4 u2/2 identically: {zero(1/Eh_alt - 1/Eh - h**4*u2/2)}", flush=True)
pt = {q[0]: 1, q[1]: 0, p[0]: 0, p[1]: 1, y: sp.Rational(1, 3)}
cols = [Z[0], Z[3], Z[4]]                                   # columns (q1, p2, y)
Jm = sp.Matrix([[sp.cancel(sp.together(sp.diff(f, v).subs(pt))) for v in cols] for f in (ch, Mh, Eh)])
minor = sp.factor(sp.cancel(Jm.det()))
print(f"    Jacobian minor of (c_h, M_h, E_h) in columns (q1, p2, y) at q=(1,0), p=(0,1), y=1/3: {minor}")
print(f"    non-zero: {minor != 0}   (total {time.time()-T0:.0f}s)")
