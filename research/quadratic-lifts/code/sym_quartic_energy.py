import sympy as sp
q, p, y, h = sp.symbols('q p y h')
z = sp.Matrix([q, p, y])
F = sp.Matrix([p, -q*y, 2*q*p])
J = F.jacobian(z)
zn = sp.simplify(z + h*(sp.eye(3) - h/2*J).LUsolve(F))
Hh = lambda Z: (Z[1]**2/2 + Z[2]**2/4)/(1 + h**2*Z[0]**2/2)
print("modified energy defect:", sp.simplify(Hh(zn) - Hh(z)))
# Jacobian determinant of the Kahan map and candidate invariant measure
Dn = sp.simplify(zn.jacobian(z).det())
print("det DPhi =", sp.factor(Dn))
