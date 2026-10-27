"""Step-size scan for the stiff open rotor chain (N=64, J=1/dx^2=100, linear frequencies up to omega_max = 2 sqrt(J) = 20).
Lifted Kahan conserves E_h and the modified circles exactly, but these do not bound the solution when h*omega is large."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
N = 64; dx = 0.1; Jc = 1/dx**2; nb = N - 1
Dm = sps.diags([-np.ones(nb), np.ones(nb)], [0, 1], shape=(nb, N)).tocsr()
def F(z):
    p, s, c = z[:N], z[N:N+nb], z[N+nb:]; th = Dm@p
    return np.concatenate([-Jc*(Dm.T@s), c*th, -s*th])
def Jac(z):
    p, s, c = z[:N], z[N:N+nb], z[N+nb:]; th = Dm@p; Z = sps.csr_matrix((N, nb))
    return sps.bmat([[sps.csr_matrix((N, N)), -Jc*Dm.T, Z], [sps.diags(c)@Dm, None, sps.diags(th)],
                     [-sps.diags(s)@Dm, -sps.diags(th), None]], format='csc')
rng = np.random.default_rng(0); th0 = 0.6*np.sin(np.linspace(0, 3*np.pi, nb)) + 0.1*rng.normal(size=nb); p0 = 0.5*rng.normal(size=N)
E = lambda p, cth: 0.5*p@p - Jc*np.sum(cth)
E0 = E(p0, np.cos(th0)); T = 20.0
print(f"omega_max = {2*np.sqrt(Jc):.0f};  Stormer-Verlet linear stability needs h < {2/(2*np.sqrt(Jc)):.2f}")
for h in (0.02, 0.05, 0.1, 0.2, 0.3):
    z = np.concatenate([p0, np.sin(th0), np.cos(th0)]); dE = 0; ok = True
    for k in range(int(T/h)):
        z = z + h*spla.spsolve((sps.identity(len(z)) - 0.5*h*Jac(z)).tocsc(), F(z))
        if not np.all(np.isfinite(z)) or np.abs(z).max() > 1e8: ok = False; break
        cth = z[N+nb:]/np.sqrt(z[N:N+nb]**2 + z[N+nb:]**2)          # cos(theta) read from the circle direction
        dE = max(dE, abs(E(z[:N], cth) - E0)/abs(E0))
    q = np.concatenate([[0.0], np.cumsum(th0)]); p = p0.copy(); dS = 0
    force = lambda q: -Jc*(Dm.T@np.sin(Dm@q))
    for k in range(int(T/h)):
        p = p + h/2*force(q); q = q + h*p; p = p + h/2*force(q); dS = max(dS, abs(E(p, np.cos(Dm@q)) - E0)/abs(E0))
    print(f"h={h:<5} (h*omega_max={h*2*np.sqrt(Jc):4.1f}): lifted Kahan {'max rel. energy error %.1e' % dE if ok else 'BLOWS UP at t=%.1f' % (k*h)};"
          f"  Stormer-Verlet max rel. energy error {dS:.1e}")
