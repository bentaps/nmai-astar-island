# Symbolic check: Kahan's method on the lifted 1-DOF system with a general quartic potential
#   z = (q, p, y),  y ~ q^2,  H~ = p^2/2 + V~(q, y)  (V~ quadratic in (q,y))
#   q' = p,  p' = -(V~_q + 2 q V~_y),  y' = 2 q p
# Claim: Kahan preserves C_h = y - q^2 + (h^2/4) p^2 exactly.
import sympy as sp

q, p, y, h = sp.symbols('q p y h')
a, b, c, d, e, f = sp.symbols('a b c d e f')
Vt = a*y**2 + b*q*y + c*y + d*q**2 + e*q + f      # general quadratic in (q, y)
Ht = p**2/2 + Vt
z = sp.Matrix([q, p, y])
F = sp.Matrix([p, -(sp.diff(Vt, q) + 2*q*sp.diff(Vt, y)), 2*q*p])
J = F.jacobian(z)
# Kahan map for quadratic fields: z' = z + h (I - h/2 F'(z))^{-1} F(z)
M = sp.eye(3) - h/2*J
zn = z + h*M.LUsolve(F)
zn = sp.simplify(zn)
C = lambda Z, P: Z[2] - Z[0]**2 + h**2/4*P**2
lhs = sp.simplify(C(zn, zn[1]) - C(z, p))
print("C_h(z') - C_h(z) =", lhs)

# Also check: is the plain (unmodified) constraint preserved? (expect: no)
print("plain constraint defect simplifies to zero?", sp.simplify((zn[2]-zn[0]**2) - (y - q**2)) == 0)
