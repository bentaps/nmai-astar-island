"""Exact (rational arithmetic) check of the Liouville-measure theorem for a mixed exponential + harmonic system in
3 DOF with a non-diagonal mass matrix: P(z') = det(DPhi(z)) P(z), P = prod u_k prod (c_j^2 + s_j^2)."""
import sympy as sp, random
R = sp.Rational
M = sp.Matrix([[2, R(1, 2), R(-1, 3)], [R(1, 2), 1, R(1, 4)], [R(-1, 3), R(1, 4), R(3, 2)]])
A = sp.Matrix([[1, R(1, 2), R(-3, 10)], [R(1, 5), -1, R(2, 5)]]); cA = [R(4, 5), R(13, 10)]
B = sp.Matrix([[R(3, 10), R(7, 10), 1], [1, R(-3, 5), R(1, 5)]]); gB = [R(11, 10), R(-1, 2)]
p = sp.symbols('p1:4'); u = sp.symbols('u1:3'); c = sp.symbols('c1:3'); s = sp.symbols('s1:3')
V = list(p) + list(u) + list(c) + list(s); v = M*sp.Matrix(p)
pd = -(A.T*sp.Matrix([cA[k]*u[k] for k in range(2)])) + B.T*sp.Matrix([gB[j]*s[j] for j in range(2)])
F = sp.Matrix(list(pd) + [u[k]*(A[k, :]*v)[0] for k in range(2)] + [-s[j]*(B[j, :]*v)[0] for j in range(2)]
              + [c[j]*(B[j, :]*v)[0] for j in range(2)])
J = F.jacobian(V); N = len(V)
P = sp.Mul(*u)*sp.Mul(*[c[j]**2 + s[j]**2 for j in range(2)])
E = (sp.Matrix(p).T*M*sp.Matrix(p))[0]/2 + sum(cA[k]*u[k] for k in range(2)) + sum(gB[j]*c[j] for j in range(2))
h = R(1, 3); random.seed(1); ok = okE = True
for trial in range(5):
    zv = [R(random.randint(-6, 6), random.randint(1, 4)) for _ in range(3)] + [R(random.randint(1, 8), random.randint(1, 4)) for _ in range(2)] \
         + [R(random.randint(-6, 6), random.randint(1, 4)) for _ in range(4)]
    sub = dict(zip(V, zv)); Mm = sp.eye(N) - h/2*J.subs(sub)
    zn = sp.Matrix(zv) + h*Mm.LUsolve(F.subs(sub)); subn = dict(zip(V, list(zn)))
    det = (sp.eye(N) + h/2*J.subs(subn)).det()/Mm.det()
    ok &= sp.simplify(P.subs(subn) - det*P.subs(sub)) == 0
    Eh = lambda sb: E.subs(sb) - h**2/8*((F[:3, 0].subs(sb)).T*M*F[:3, 0].subs(sb))[0]
    okE &= sp.simplify(Eh(subn) - Eh(sub)) == 0
print("mixed exp + cos, 3 DOF, non-diagonal M, exact rational check at 5 points: measure", ok, "; modified energy", okE)
