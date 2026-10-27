"""Open rotor (XY) chain, H = |p|^2/2 - J sum_k cos(q_{k+1} - q_k), lifted with (s_k, c_k) = (sin, cos) of the bond
angles: p_k' = J (s_k - s_{k-1}) (linear in s), s_k' = c_k (p_{k+1} - p_k), c_k' = -s_k (p_{k+1} - p_k).
Lemma: E_h = E - (h^2/8)|p'|^2 is conserved; two-variable theorem: (s_k^2 + c_k^2)/(1 + h^2 theta_k'^2/4) conserved.
Stiff version J = 1/dx^2: lifted Kahan runs stably at 3x the Stormer-Verlet stability limit, Stormer-Verlet blows up."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
N = 64; dx = 0.1; Jc = 1/dx**2; nb = N - 1
Dm = sps.diags([-np.ones(nb), np.ones(nb)], [0, 1], shape=(nb, N)).tocsr()      # theta' = D p
def F(z):
    p, s, c = z[:N], z[N:N+nb], z[N+nb:]; th = Dm@p
    return np.concatenate([-Jc*(Dm.T@s), c*th, -s*th])
def Jac(z):
    p, s, c = z[:N], z[N:N+nb], z[N+nb:]; th = Dm@p; Z = sps.csr_matrix((N, nb))
    return sps.bmat([[sps.csr_matrix((N, N)), -Jc*Dm.T, Z], [sps.diags(c)@Dm, None, sps.diags(th)],
                     [-sps.diags(s)@Dm, -sps.diags(th), None]], format='csc')
def kahan(z, h): return z + h*spla.spsolve((sps.identity(len(z)) - 0.5*h*Jac(z)).tocsc(), F(z))
rng = np.random.default_rng(0); th0 = 0.6*np.sin(np.linspace(0, 3*np.pi, nb)) + 0.1*rng.normal(size=nb); p0 = 0.5*rng.normal(size=N)
z = np.concatenate([p0, np.sin(th0), np.cos(th0)])
lam_max = 4*Jc; h_sv = 2/np.sqrt(lam_max); h = 3*h_sv                       # Stormer-Verlet limit ~ 2/sqrt(max eigenvalue)
E = lambda z: 0.5*z[:N]@z[:N] - Jc*np.sum(z[N+nb:])
Eh = lambda z: E(z) - h**2/8*F(z)[:N]@F(z)[:N]
circ = lambda z: (z[N:N+nb]**2 + z[N+nb:]**2)/(1 + h**2*(Dm@z[:N])**2/4)
E0, Eh0, C0 = E(z), Eh(z), circ(z); dE = dEh = dC = 0
for k in range(2000):
    z = kahan(z, h); dEh = max(dEh, abs(Eh(z) - Eh0)/abs(Eh0)); dC = max(dC, np.abs(circ(z) - C0).max()); dE = max(dE, abs(E(z) - E0)/abs(E0))
print(f"rotor chain N={N}, J=1/dx^2={Jc:.0f}, h = 3 x Stormer-Verlet limit = {h:.3f}, 2000 steps:")
print(f"   lifted Kahan: drift of E_h {dEh:.1e}, of the circles {dC:.1e}; original energy error {dE:.1e}")
# Stormer-Verlet on the original angles at the same h
q = np.concatenate([[0.0], np.cumsum(th0)]); p = p0.copy()
force = lambda q: -Jc*(Dm.T@np.sin(Dm@q))
blew = False
for k in range(2000):
    p = p + h/2*force(q); q = q + h*p; p = p + h/2*force(q)
    if not np.all(np.isfinite(p)) or np.abs(p).max() > 1e6: blew = True; break
dS = 0
print(f"   Stormer-Verlet at the same h: {'blows up after ' + str(k+1) + ' steps' if blew else 'no overflow (see rotor_chain_scan.py for its energy error)'}")
