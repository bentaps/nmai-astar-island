"""1-DOF family H = U(phi) + kappa: Kahan preserves the measure dz / (D_l1 D_l2), the product of the two CMOQ
denominators D_l = 1 + (h^2/4) mu_l u2 (l_l^T J Phi x)^2 of the pencil members. This is what CMOQ 2019, Theorem 2
predicts after the change of variables (w1, y, w2) given by the pencil; here it is checked numerically."""
import numpy as np
Jc = np.array([[0., 1.], [-1., 0.]])
def setup(Phi, K, u2, u1):
    def F(z):
        x, y = z[:2], z[2]; Up = u2*y + u1
        g = K@x + Up*(Phi@x)
        return np.concatenate([Jc@g, [(Phi@x)@(Jc@(K@x))]])
    def JF(z):
        I3 = np.eye(3); return np.array([(F(z + I3[k]) - F(z - I3[k]))/2 for k in range(3)]).T
    # pencil roots / rank-one factors
    a = np.linalg.det(Phi); b = -(Phi[0,0]*K[1,1] + Phi[1,1]*K[0,0] - 2*Phi[0,1]*K[0,1]); c = np.linalg.det(K)
    dens = []
    for lam in np.roots([a, b, c]):
        M = K - lam*Phi
        if abs(M[0, 0]) > 1e-12: l = np.array([1, M[0, 1]/M[0, 0]]); mu = M[0, 0]
        else: l = np.array([M[0, 1]/M[1, 1], 1]); mu = M[1, 1]
        dens.append((mu, l))
    return F, JF, dens
def test(Phi, K, u2, u1, h=0.15, label=""):
    F, JF, dens = setup(np.array(Phi, float), np.array(K, float), u2, u1)
    Phi = np.array(Phi, float)
    def P(z):
        x = z[:2]; val = 1.0
        for mu, l in dens: val *= 1 + h**2/4*mu*u2*(l@Jc@Phi@x)**2
        return val
    rng = np.random.default_rng(0); worst = 0
    for _ in range(300):
        z = rng.uniform(-1, 1, 3)
        M = np.eye(3) - h/2*JF(z); zn = z + h*np.linalg.solve(M, F(z))
        det = np.linalg.det(np.linalg.solve(M, np.eye(3) + h/2*JF(zn)))
        worst = max(worst, abs(P(zn) - det*P(z))/abs(P(z)))
    print(f"{label:28s} max relative defect of P(z') = det(DPhi) P(z): {worst:.1e}")
test([[1, .3], [.3, .6]], [[.7, -.2], [-.2, 1.1]], 1.0, 0.2, label="random real distinct roots")
test([[1.4, -.5], [-.5, .8]], [[.5, .1], [.1, .9]], 0.6, -0.3, label="another random instance")
test([[0, 1], [1, 0]], [[1, 0], [0, 1]], 1.0, 0.0, label="mixed example")
