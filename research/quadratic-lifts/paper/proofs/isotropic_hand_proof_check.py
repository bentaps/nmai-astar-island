"""Checks each algebraic step of the hand proofs of Theorem 6.1(b),(c) in the paper (E_h and the measure), with
sympy, for n = 3 and symbolic h, u1, u2. Each line prints True if the step holds identically.
Every quantity checked is rotation invariant (or, for the vector identities, rotation equivariant), so it suffices
to start from the slice q = (q1, 0, 0), p = (p1, p2, 0); the image point is not restricted."""
import sympy as sp
n = 3
h, u2, u1 = sp.symbols('h u2 u1')
q1, p1, p2, y = sp.symbols('q1 p1 p2 y')
q = sp.Matrix([q1, 0, 0]); p = sp.Matrix([p1, p2, 0])
Yt = y + u1/u2                                       # shifted auxiliary variable, U'(y) = u2*Yt
kap = h**2*u2
def step(q, p, y):                                   # Kahan's map (closed form, checked in isotropic_proof.py)
    al = h*(u2*y + u1); Yf = 1 + h*al/2; a = q.dot(q); b = q.dot(p); c = p.dot(p)
    yn = (y*Yf - h*(al + h*u1)*(a + h*b/2) + 2*h*b + h**2*c)/(Yf + h**2*u2*(a + h*b/2))
    aln = h*(u2*yn + u1)
    return ((1 - h*aln/2)*q + h*p)/Yf, (-(al + aln)*q + (1 - h*al/2)*p)/Yf, yn
qn, pn, yn = step(q, p, y)
Ytn = yn + u1/u2
Y, Yn = 1 + h**2*u2*Yt/2, 1 + h**2*u2*Ytn/2
K = p.dot(p)/2 + u2*Yt**2/2; Kn = pn.dot(pn)/2 + u2*Ytn**2/2
def wedge2(a, b): return a.dot(a)*b.dot(b) - a.dot(b)**2      # |a ^ b|^2 (Lagrange's identity)
Z = lambda e: sp.cancel(sp.together(e)) == 0
print("(K1) q'-q = h/2 (p+p'):", all(Z(e) for e in qn - q - h/2*(p + pn)))
# ---- E_h: crossed identity from the block (p, Ytilde) with speeds q
lhs = Kn*(1 + kap*q.dot(q)) - kap/2*wedge2(q, pn)
rhs = K*(1 + kap*qn.dot(qn)) - kap/2*wedge2(qn, p)
print("(crossed identity, Sec. 6) K'(1+k|q|^2) - k/2|q^p'|^2 = K(1+k|q'|^2) - k/2|q'^p|^2:", Z(lhs - rhs))
Mh2 = wedge2(q, p)/Y**2
print("|q^p'| = (2-Y)|M_h|  and  |q'^p| = (2-Y')|M_h|:", Z(wedge2(q, pn) - (2 - Y)**2*Mh2), Z(wedge2(qn, p) - (2 - Yn)**2*Mh2))
D = 1 + kap*q.dot(q) - h**2*kap/2*K; Dn = 1 + kap*qn.dot(qn) - h**2*kap/2*Kn
Eh = (K + kap/2*Mh2)/D; Ehn = (Kn + kap/2*wedge2(qn, pn)/Yn**2)/Dn
print("E_h(z') = E_h(z):", Z(Ehn - Eh))
# ---- measure: xi, eta, eta' (named v, w, wn below)
w = q + h/2*p; v = q - h/2*p; wn = qn + h/2*pn
print("eta = q' - h/2 p':", all(Z(e) for e in w - (qn - h/2*pn)))
A, B, C = v.dot(w), w.dot(wn), w.dot(w)
print("(R1) Y' - Y = k/2 (B - A):", Z(Yn - Y - kap/2*(B - A)))
print("(K2') Y eta' + Y' xi = (4 - Y - Y') eta:", all(Z(e) for e in Y*wn + Yn*v - (4 - Y - Yn)*w))
print("(R2) Y B + Y' A = (4 - Y - Y') C:", Z(Y*B + Yn*A - (4 - Y - Yn)*C))
print("D = (2-Y) Y + k A,  D' = (2-Y') Y' + k B:", Z(D - ((2 - Y)*Y + kap*A)), Z(Dn - ((2 - Yn)*Yn + kap*B)))
Zs = sp.Matrix(sp.symbols('Q1:4 P1:4 Ys'))               # generic point for the Jacobian
Qs, Ps, Ys = Zs[:3, 0], Zs[3:6, 0], Zs[6]
Fz = sp.Matrix([*Ps, *(-2*(u2*Ys + u1)*Qs), 2*Qs.dot(Ps)])
Jf = Fz.jacobian(Zs)
detm = (sp.eye(2*n + 1) - h/2*Jf).det().subs(dict(zip(Zs, [*q, *p, y])), simultaneous=True)
print("det(I - h/2 F'(z)) = Y^(n-1) (Y + k/2 (A + C)):", Z(detm - Y**(n - 1)*(Y + kap/2*(A + C))))
detp = (sp.eye(2*n + 1) + h/2*Jf).det().subs(dict(zip(Zs, [*qn, *pn, yn])), simultaneous=True)
print("det(I + h/2 F'(z')) = Y'^(n-1) (Y' + k/2 (B + C)):", Z(detp - Yn**(n - 1)*(Yn + kap/2*(B + C))))
print("2D = (4 - Y - Y')(Y + k/2 (A + C))  and  2D' = (4 - Y - Y')(Y' + k/2 (B + C)):",
      Z(2*D - (4 - Y - Yn)*(Y + kap/2*(A + C))), Z(2*Dn - (4 - Y - Yn)*(Yn + kap/2*(B + C))))
