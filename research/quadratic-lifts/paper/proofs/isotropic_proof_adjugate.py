"""Independent cross-check (slower) of Theorem 6.1(b),(c) for n = 2: the Kahan step is computed from the
adjugate formula z' = z + h adj(M) F(z) / det(M), M = I - h/2 F'(z), instead of the closed form, and the
identities are checked on the slice q2 = 0 (rotation invariance). Uses the alternative form of the modified energy,
E_alt = (K + h^2 u2 |M_h|^2/2) / (1 + h^2 u2 |q|^2 + h^6 u2^2 |M_h|^2/4), with 1/E_alt = 1/E_h + h^4 u2/2.
Recorded runs: E_alt with symbolic u1, u2 in 404 s; the measure identity for u2 = 1/2, u1 = 0 in 567 s (the measure
with symbolic u1, u2 takes hours this way; isotropic_proof.py does it in seconds).
Usage: python3 isotropic_proof_adjugate.py [u2 u1]   (default: symbolic u2, u1)."""
import sympy as sp, sys, time
q1, p1, p2, y, h = sp.symbols('q1 p1 p2 y h')
if len(sys.argv) == 3: u2, u1 = sp.Rational(sys.argv[1]), sp.Rational(sys.argv[2])
else: u2, u1 = sp.symbols('u2 u1')
Q = [q1, sp.Integer(0)]                                    # q = (q1, 0) after a rotation
Z = sp.Matrix([*Q, p1, p2, y]); qq = sp.symbols('Q1 Q2'); pp = sp.symbols('P1 P2'); yy = sp.Symbol('Y')
Zs = sp.Matrix([*qq, *pp, yy])
Up = lambda Y: u2*Y + u1                                    # U'(y)
Fs = sp.Matrix([pp[0], pp[1], -2*Up(yy)*qq[0], -2*Up(yy)*qq[1], 2*(qq[0]*pp[0] + qq[1]*pp[1])])
Js = Fs.jacobian(Zs)
sub0 = dict(zip(Zs, Z))
M = sp.eye(5) - h/2*Js.subs(sub0); Fz = Fs.subs(sub0)
t0 = time.time()
detM = sp.factor(M.det()); adj = M.adjugate()
Zn = [sp.together(Z[i] + h*sum(adj[i, j]*Fz[j] for j in range(5))/detM) for i in range(5)]
print(f"Kahan step computed symbolically ({time.time()-t0:.0f}s)", flush=True)
def invariants(z):
    a1, a2, b1, b2, Y = z
    Q2, P2, Mv = a1**2 + a2**2, b1**2 + b2**2, a1*b2 - a2*b1
    Mh = Mv/(1 + h**2*Up(Y)/2)
    ch = Y - Q2 + h**2*P2/4
    Eh = (P2/2 + u2*Y**2/2 + u1*Y + u1**2/(2*u2) + h**2*u2/2*Mh**2)/(1 + h**2*u2*Q2 + h**6*u2**2/4*Mh**2)
    P = (1 + h**2*Up(Y)/2)*(1 + h**2*u2*Q2 - h**4*u2/2*(P2/2 + u2*Y**2/2 + u1*Y + u1**2/(2*u2)))
    return {"c_h": ch, "M_h": Mh, "E_h": Eh}, P
I0, P0 = invariants(list(Z)); I1, P1 = invariants(Zn)
for k in I0:
    t0 = time.time(); ok = sp.cancel(sp.together(I1[k] - I0[k])) == 0
    print(f"{k}(z') - {k}(z) == 0 identically: {ok} ({time.time()-t0:.0f}s)", flush=True)
# measure: P(z') == det(DPhi(z)) P(z), det(DPhi) = det(I + h/2 J(z')) / det(I - h/2 J(z))
t0 = time.time()
Mp = sp.eye(5) + h/2*Js.subs(dict(zip(Zs, Zn)))
detMp = sp.together(Mp.det())
ok = sp.cancel(sp.together(P1*detM - detMp*P0)) == 0
print(f"measure identity P(z') = det(DPhi) P(z) holds identically: {ok} ({time.time()-t0:.0f}s)", flush=True)
