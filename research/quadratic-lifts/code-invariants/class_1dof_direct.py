"""Direct one-step verification of the predicted modified integrals for the 1-DOF class
H = U(phi(x)) + K(x).  For each root lam of det(K - lam Phi) = 0:  K - lam Phi = mu * l l^T (rank 1), and
   I_lam = Ht + lam c = (mu/2) w^2 + (u2/2) y^2 + (u1 + lam) y,   w = l.x   (two variables (w, y)),
   A_lam = w' / (u2 y + u1 + lam)   (affine; computed by polynomial division),
   CMOQ:  I~_lam = (I_lam + h^2 D2 A^2/8) / (1 + h^2 D1 A^2/4),  D1 = mu u2,  D2 = -mu (u1 + lam)^2."""
import numpy as np, sympy as sp
q, p, y = sp.symbols('q p y')
def check(Phi, K, u2, u1, h=sp.Rational(1, 7), npts=5, seed=0):
    Phi, K = sp.Matrix(Phi), sp.Matrix(K)
    x = sp.Matrix([q, p]); phi = (x.T*Phi*x)[0]/2; Kf = (x.T*K*x)[0]/2
    U = u2*y**2/2 + u1*y
    F = sp.Matrix([sp.diff(Kf, p) + sp.diff(phi, p)*sp.diff(U, y),
                   -(sp.diff(Kf, q) + sp.diff(phi, q)*sp.diff(U, y)),
                   sp.diff(phi, q)*sp.diff(Kf, p) - sp.diff(phi, p)*sp.diff(Kf, q)])
    z = sp.Matrix([q, p, y]); J = F.jacobian(z)
    lam = sp.symbols('lam'); roots = sp.solve(sp.Eq((K - lam*Phi).det(), 0), lam)
    out = []
    rng = np.random.default_rng(seed)
    for r in roots:
        M = K - r*Phi
        # rank-1 factorisation M = mu l l^T
        if M[0, 0] != 0: l = sp.Matrix([1, M[0, 1]/M[0, 0]]); mu = M[0, 0]
        else: l = sp.Matrix([M[0, 1]/M[1, 1], 1]); mu = M[1, 1]
        w = (l.T*x)[0]
        I = mu*w**2/2 + u2*y**2/2 + (u1 + r)*y
        wdot = sp.expand((l.T*F[:2, :])[0])
        A, rem = sp.div(sp.Poly(wdot, q, p, y), sp.Poly(u2*y + u1 + r, q, p, y))
        assert rem.is_zero, "A not affine?"
        A = A.as_expr(); D1 = mu*u2; D2 = -mu*(u1 + r)**2
        It = (I + h**2*D2*A**2/8)/(1 + h**2*D1*A**2/4)
        It_f = sp.lambdify((q, p, y), It, 'mpmath')
        maxdev = 0
        for _ in range(npts):
            zv = [sp.Rational(int(rng.integers(-9, 10)), int(rng.integers(1, 7))) for _ in range(3)]
            sub = dict(zip((q, p, y), zv))
            zn = sp.Matrix(zv) + h*((sp.eye(3) - h/2*J.subs(sub)).LUsolve(F.subs(sub)))
            d = sp.nsimplify(sp.simplify(It.subs(dict(zip((q, p, y), list(zn)))) - It.subs(sub)))
            maxdev = max(maxdev, abs(complex(sp.N(d, 30))))
        out.append((sp.nsimplify(r), maxdev))
    return out
R = sp.Rational
print("real distinct :", check([[1, R(3,10)], [R(3,10), R(3,5)]], [[R(7,10), -R(1,5)], [-R(1,5), R(11,10)]], 1, R(1,5)))
print("complex pair  :", check([[1, 0], [0, -1]], [[R(3,10), 1], [1, -R(3,10)]], R(4,5), R(1,10)))
print("mixed example :", check([[0, 1], [1, 0]], [[1, 0], [0, 1]], 1, 0))
