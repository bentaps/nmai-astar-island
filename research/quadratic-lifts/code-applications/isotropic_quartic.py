"""A new integrable map: Kahan's method on the lift y = |q|^2 of the isotropic quartic oscillator
H = |p|^2/2 + |q|^4/4 in R^2 (lifted field q' = p, p' = -y q, y' = 2 q.p; state z in R^5).
Integrals (all exact, for every h):
   c_h = y - |q|^2 + h^2 |p|^2/4                                   (companion lemma)
   M_h = (q1 p2 - q2 p1)/(1 + h^2 y/4)                             (block theorem with two speeds)
   E_h = (|p|^2/2 + y^2/4 + h^2 M_h^2/4)/(1 + h^2 |q|^2/2 + h^6 M_h^2/16)   (found by a Darboux-polynomial search)
and the measure dz / [(1 + h^2 y/4)(1 + h^2|q|^2/2 - h^4 (|p|^2/2 + y^2/4)/4)].
Also: modified angular momenta M_ab/(1 + h^2 U'(y)/2) for the isotropic quartic in R^3 with U = u2 y^2/2 + u1 y."""
import numpy as np
from qlift import kahan_step, run

def jac(F, N):
    I = np.eye(N); return lambda z: np.array([(F(z + I[k]) - F(z - I[k]))/2 for k in range(N)]).T

def F(z):
    q, p, y = z[:2], z[2:4], z[4]; return np.concatenate([p, -y*q, [2*q@p]])
JF = jac(F, 5)
for h in (0.3, 0.5, 0.9):
    rng = np.random.default_rng(5); worst = {}
    for t in range(6):
        q0, p0 = rng.uniform(-0.9, 0.9, 2), rng.uniform(-0.9, 0.9, 2)
        off = 0.0 if t < 3 else rng.uniform(-0.3, 0.3)             # three orbits on, three off the shifted sheet
        z0 = np.concatenate([q0, p0, [q0@q0 - h**2/4*p0@p0 + off]])
        Z = run(lambda z: kahan_step(F, JF, z, h), z0, 20000)
        q, p, y = Z[:, :2], Z[:, 2:4], Z[:, 4]; Q2, P2 = (q**2).sum(1), (p**2).sum(1)
        M = q[:, 0]*p[:, 1] - q[:, 1]*p[:, 0]; Mh = M/(1 + h**2*y/4)
        Eh = (P2/2 + y**2/4 + h**2*Mh**2/4)/(1 + h**2*Q2/2 + h**6*Mh**2/16)
        checks = [("c_h", y - Q2 + h**2/4*P2), ("M_h", Mh), ("E_h", Eh)]
        if t < 3: checks.append(("original energy (on-sheet orbits)", P2/2 + Q2**2/4))
        for name, v in checks:
            worst[name] = max(worst.get(name, 0), abs(v - v[0]).max()/max(1.0, abs(v[0])))
    # measure: density 1/P with P = (1 + h^2 y/4)(1 + h^2|q|^2/2 - h^4 E~/4), check P(z') = det(DPhi) P(z)
    Pd = lambda z: (1 + h**2*z[4]/4)*(1 + h**2*(z[:2]@z[:2])/2 - h**4/4*(z[2:4]@z[2:4]/2 + z[4]**2/4))
    dmax = 0
    for _ in range(200):
        z = rng.uniform(-1, 1, 5); Mm = np.eye(5) - h/2*JF(z); zn = z + h*np.linalg.solve(Mm, F(z))
        det = np.linalg.det(np.linalg.solve(Mm, np.eye(5) + h/2*JF(zn))); dmax = max(dmax, abs(Pd(zn) - det*Pd(z))/abs(Pd(z)))
    print(f"h={h}: max drift over 6 orbits x 2e4 steps: " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items())
          + f";  measure defect {dmax:.1e}")

# modified angular momenta in R^3 with a general U(y) = u2 y^2/2 + u1 y
n, u2, u1, h = 3, 0.7, 0.4, 0.3
def F3(z):
    q, p, y = z[:n], z[n:2*n], z[2*n]; return np.concatenate([p, -2*(u2*y + u1)*q, [2*q@p]])
rng = np.random.default_rng(1); q0 = rng.normal(size=n)*0.8; p0 = rng.normal(size=n)*0.5
Z = run(lambda z: kahan_step(F3, jac(F3, 2*n + 1), z, h), np.concatenate([q0, p0, [q0@q0]]), 50000)
q, p, y = Z[:, :n], Z[:, n:2*n], Z[:, 2*n]; worst = 0
for a in range(n):
    for b in range(a + 1, n):
        Mh = (q[:, a]*p[:, b] - q[:, b]*p[:, a])/(1 + h**2/2*(u2*y + u1)); worst = max(worst, abs(Mh - Mh[0]).max()/abs(Mh[0]))
print(f"isotropic quartic in R^3 (u2={u2}, u1={u1}, h={h}, 5e4 steps): max drift of M_ab/(1 + h^2 U'(y)/2) = {worst:.1e}")
