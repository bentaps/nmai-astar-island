"""Long-time behaviour of lifted Kahan on the stiff rotor chain: when do the unbounded modified circles let the
lifted variables run away?"""
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
E = lambda p, cth: 0.5*p@p - Jc*np.sum(cth); E0 = E(p0, np.cos(th0))
for h in (0.05, 0.1, 0.2, 0.3):
    z = np.concatenate([p0, np.sin(th0), np.cos(th0)]); T = 600.0; marks = {}; Eh0 = None
    Eh = lambda z: 0.5*z[:N]@z[:N] - Jc*np.sum(z[N+nb:]) - h**2/8*F(z)[:N]@F(z)[:N]; Eh0 = Eh(z); dEh = 0
    for k in range(int(T/h)):
        z = z + h*spla.spsolve((sps.identity(len(z)) - 0.5*h*Jac(z)).tocsc(), F(z))
        t = (k + 1)*h; rad = np.sqrt(z[N:N+nb]**2 + z[N+nb:]**2).max()
        if not np.isfinite(rad) or rad > 1e6: marks['blow-up'] = t; break
        dEh = max(dEh, abs(Eh(z) - Eh0)/abs(Eh0))
        for tm in (20, 100, 300, 600):
            if abs(t - tm) < h/2: marks[tm] = (rad, abs(E(z[:N], z[N+nb:]/np.sqrt(z[N:N+nb]**2 + z[N+nb:]**2)) - E0)/abs(E0))
    print(f"h={h:<5} (h*omega={2*np.sqrt(Jc)*h:3.1f}): " + "; ".join(
        (f"t={k}: max sqrt(s^2+c^2)={v[0]:.2f}, energy err {v[1]:.1e}" if k != 'blow-up' else f"BLOW-UP at t={v:.0f}") for k, v in marks.items())
        + f"; E_h drift before blow-up {dEh:.1e}", flush=True)
