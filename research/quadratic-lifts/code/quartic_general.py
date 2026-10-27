import numpy as np
from qlift import kahan_step, run, algebraic_relation_spectrum
np.set_printoptions(precision=3, linewidth=150)
# V(q) = q^4/4 + q^3/3 + q^2/2  (non-symmetric; single well)
V = lambda q: q**4/4 + q**3/3 + q**2/2
H = lambda q, p: p**2/2 + V(q)

def make_lift(b):
    """Lift y = q^2 + b q. Write V = U(y) + kappa*q with U quadratic: V = a2 y^2 + a1 y + kappa q."""
    # q^4/4 + q^3/3 + q^2/2 = a2 (q^2+bq)^2 + a1 (q^2+bq) + kappa q
    a2 = 0.25
    # q^3: 2 a2 b = 1/3 -> only consistent if b = 2/3; for other b keep a q^3 term via q*y
    # general: V~(q,y) = a2 y^2 + c3 q y + a1 y + c1 q  with consistency on y = q^2 + b q
    # expand: a2(q^4 + 2b q^3 + b^2 q^2) + c3(q^3 + b q^2) + a1(q^2 + b q) + c1 q
    c3 = 1/3 - 2*a2*b
    a1 = 0.5 - a2*b**2 - c3*b
    c1 = -a1*b
    Vt = lambda q, y: a2*y**2 + c3*q*y + a1*y + c1*q
    dVq = lambda q, y: c3*y + c1          # partial wrt q
    dVy = lambda q, y: 2*a2*y + c3*q + a1  # partial wrt y
    # lifted ODE: q' = p, p' = -(dVq + (2q+b) dVy), y' = (2q+b) p
    F = lambda z: np.array([z[1], -(dVq(z[0], z[2]) + (2*z[0]+b)*dVy(z[0], z[2])), (2*z[0]+b)*z[1]])
    def JF(z):
        q, p, y = z
        dpdq = -(0 + 2*dVy(q, y) + (2*q+b)*c3)
        dpdy = -(c3 + (2*q+b)*2*a2)
        return np.array([[0, 1, 0], [dpdq, 0, dpdy], [2*p, 0, 0]]) + np.array([[0,0,0],[0,0,0],[0,2*q+b,0]])
    return F, JF, Vt, (a2, c3, a1, c1)

h = 0.1
for b in [0.0, 2/3]:
    F, JF, Vt, coef = make_lift(b)
    print(f"\n=== lift y = q^2 + {b:.4f} q ;  V~ = {coef[0]} y^2 + {coef[1]:.4f} q y + {coef[2]:.4f} y + {coef[3]:.4f} q")
    q0, p0 = 0.8, 0.3
    z0 = np.array([q0, p0, q0**2 + b*q0])
    Z = run(lambda z: kahan_step(F, JF, z, h), z0, 3000)
    Ct = Z[:, 2] - Z[:, 0]**2 - b*Z[:, 0] + h**2/4*Z[:, 1]**2
    print("  modified constraint drift:", np.max(abs(Ct - Ct[0])))
    for deg in [2, 3, 4, 6, 8]:
        s, _ = algebraic_relation_spectrum(Z, deg)
        print(f"  deg {deg}: #monomials {len(s)}  #sv<1e-10: {np.sum(s<1e-10)}  (curve-on-quadric count if only C~: {len(s)-len(algebraic_relation_spectrum(Z[:, :1]*0+1, 0)[0])})" if False else
              f"  deg {deg}: #monomials {len(s)}  #sv<1e-10: {np.sum(s<1e-10)}  next sv: {s[len(s)-np.sum(s<1e-10)-1]:.2e}")
    Zl = run(lambda z: kahan_step(F, JF, z, h), z0, 400000)
    E = H(Zl[:, 0], Zl[:, 1])
    print(f"  long run (T=4e4): max|H-H0| first 10%: {np.max(abs(E[:40000]-E[0])):.3e}, last 10%: {np.max(abs(E[-40000:]-E[0])):.3e}")
