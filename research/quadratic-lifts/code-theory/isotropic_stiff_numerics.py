"""New numerics: the lifted Kahan map for the stiff isotropic anharmonic oscillator in R^3,
H = |p|^2/2 + om^2 |q|^2/2 + |q|^4/4  (U(y) = y^2/4 + om^2 y/2, lift y = |q|^2).
1. The map keeps q', p' in span{q, p} (planar motion), so any dimension reduces to n = 2.
2. Stiff case: lifted Kahan (one 7x7 linear solve per step) conserves c_h, all M_ab,h and E_h exactly at
   h*om = 5, where Stormer-Verlet is unstable. Compared with the implicit midpoint rule (Newton) on (q, p)."""
import numpy as np, time
n = 3; om = 10.0; u2, u1 = 0.5, om**2/2
def F(z):
    q, p, y = z[:n], z[n:2*n], z[2*n]; return np.concatenate([p, -2*(u2*y + u1)*q, [2*q@p]])
def JF(z):
    q, p, y = z[:n], z[n:2*n], z[2*n]; J = np.zeros((2*n + 1, 2*n + 1))
    J[:n, n:2*n] = np.eye(n); J[n:2*n, :n] = -2*(u2*y + u1)*np.eye(n); J[n:2*n, 2*n] = -2*u2*q
    J[2*n, :n] = 2*p; J[2*n, n:2*n] = 2*q; return J
def kahan(z, h): return z + h*np.linalg.solve(np.eye(2*n + 1) - h/2*JF(z), F(z))
H = lambda q, p: p@p/2 + om**2*(q@q)/2 + (q@q)**2/4
# 1. plane preservation (n = 4 random point)
rng = np.random.default_rng(0); n4 = 4
def F4(z):
    q, p, y = z[:4], z[4:8], z[8]; return np.concatenate([p, -2*(u2*y + 1.0)*q, [2*q@p]])
I9 = np.eye(9); J4 = lambda z: np.array([(F4(z + I9[k]) - F4(z - I9[k]))/2 for k in range(9)]).T
z = rng.normal(size=9); zn = z + 0.7*np.linalg.solve(np.eye(9) - 0.35*J4(z), F4(z))
sv = np.linalg.svd(np.column_stack([z[:4], z[4:8], zn[:4], zn[4:8]]), compute_uv=False)
print(f"n=4: singular values of [q, p, q', p'] = {np.round(sv, 12)}  -> q', p' lie in span{{q, p}}")
# 2. stiff test
q0 = np.array([0.8, 0.1, -0.3]); p0 = np.array([0.0, 3.0, 1.0]); T = 200.0
for h in (0.05, 0.5):
    z = np.concatenate([q0, p0, [q0@q0 - h**2/4*p0@p0]]); E0 = H(q0, p0); M0 = np.cross(q0, p0)
    Mh = lambda z: np.cross(z[:n], z[n:2*n])/(1 + h**2/2*(u2*z[2*n] + u1))
    def Eh(z):
        q, p, y = z[:n], z[n:2*n], z[2*n]; Mh2 = Mh(z)@Mh(z)
        return (p@p/2 + u2*y**2/2 + u1*y + u1**2/(2*u2) + h**2*u2/2*Mh2)/(1 + h**2*u2*(q@q) + h**6*u2**2/4*Mh2)
    Eh0, Mh0 = Eh(z), Mh(z); dE = dM = dEh = dMh = 0; t0 = time.time()
    for k in range(int(T/h)):
        z = kahan(z, h); q, p = z[:n], z[n:2*n]
        dE = max(dE, abs(H(q, p) - E0)/E0); dM = max(dM, np.abs(np.cross(q, p) - M0).max()/np.abs(M0).max())
        dEh = max(dEh, abs(Eh(z) - Eh0)/abs(Eh0)); dMh = max(dMh, np.abs(Mh(z) - Mh0).max()/np.abs(Mh0).max())
    tk = time.time() - t0
    # implicit midpoint on (q, p) with Newton
    q, p = q0.copy(), p0.copy(); dEm = 0; its = 0; t0 = time.time()
    for k in range(int(T/h)):
        Q, P = q.copy(), p.copy()
        for it in range(50):
            qm, pm = (q + Q)/2, (p + P)/2; r2 = qm@qm
            G = np.concatenate([Q - q - h*pm, P - p + h*(om**2 + r2)*qm])
            Jg = np.block([[np.eye(n), -h/2*np.eye(n)], [h/2*((om**2 + r2)*np.eye(n) + 2*np.outer(qm, qm)), np.eye(n)]])
            d = np.linalg.solve(Jg, -G); Q += d[:n]; P += d[n:]; its += 1
            if np.abs(d).max() < 1e-14: break
        q, p = Q, P; dEm = max(dEm, abs(H(q, p) - E0)/E0)
    tm = time.time() - t0
    # Stormer-Verlet
    q, p = q0.copy(), p0.copy(); blow = False
    for k in range(int(T/h)):
        p = p - h/2*(om**2 + q@q)*q; q = q + h*p; p = p - h/2*(om**2 + q@q)*q
        if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6: blow = True; break
    dES = np.nan if blow else abs(H(q, p) - E0)/E0
    print(f"h*om={h*om:.1f}: lifted Kahan: E_h drift {dEh:.1e}, M_h drift {dMh:.1e}, max energy err {dE:.1e}, max ang.mom err {dM:.1e}"
          f" ({int(T/h)} linear solves, {tk:.2f}s) | midpoint: energy err {dEm:.1e} ({its} Newton solves, {tm:.2f}s)"
          f" | Stormer-Verlet: {'blows up' if blow else 'final energy err %.1e' % dES}")
