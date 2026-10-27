"""Isotropic quartic in a uniform magnetic field, lifted (q' = v, v' = J v - y q, y' = 2 q.v, omega = 1), Kahan with
h = 1/5. An exact Darboux search (magnetic_search.py) finds two densities 1/P1, 1/P2 (degrees 3 and 4); their ratio
I = P1/P2 is a rational integral mixing M = q x v with energy-like terms. Check it numerically along orbits."""
import numpy as np
from qlift import kahan_step, run
def F(z):
    q1, q2, v1, v2, y = z; return np.array([v1, v2, -v2 - y*q1, v1 - y*q2, 2*(q1*v1 + q2*v2)])
I5 = np.eye(5); JF = lambda z: np.array([(F(z + I5[k]) - F(z - I5[k]))/2 for k in range(5)]).T
def P1(z):
    q1, q2, v1, v2, y = z; Q2, V2, M = q1**2 + q2**2, v1**2 + v2**2, q1*v2 - q2*v1
    return -200*Q2*y - 20000*Q2 - 400*M + 2*V2*y + 200*V2 + y**3 + 100*y**2 - 9900*y - 1010000
def P2(z):
    q1, q2, v1, v2, y = z; Q2, V2, M = q1**2 + q2**2, v1**2 + v2**2, q1*v2 - q2*v1
    return M*(200*Q2 - 2*V2 - y**2 + 9900) - (200*Q2*y + 10000*Q2 + 100*V2 + 50*y**2 + 10000*y + 505000)
h = 0.2; rng = np.random.default_rng(2)
for t in range(3):
    q0, v0 = rng.uniform(-0.8, 0.8, 2), rng.uniform(-0.8, 0.8, 2)
    Z = run(lambda z: kahan_step(F, JF, z, h), np.concatenate([q0, v0, [q0@q0 - h**2/4*v0@v0]]), 20000)
    I = np.array([P1(z)/P2(z) for z in Z]); M = Z[:, 0]*Z[:, 3] - Z[:, 1]*Z[:, 2]
    print(f"orbit {t}: drift of P1/P2 = {abs(I - I[0]).max()/abs(I[0]):.1e};  drift of plain q x v = {abs(M - M[0]).max():.1e}")
