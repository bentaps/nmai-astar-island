import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
from qlift import kahan_step, run
def jac(F, N):
    I = np.eye(N)
    return lambda z: np.array([(F(z + I[k]) - F(z - I[k]))/2 for k in range(N)]).T
# ---- octic chain robustness
Fo = lambda z: np.array([z[1], -z[4]*z[3], 2*z[0]*z[1], 4*z[4]*z[1], 3*z[2]*z[1]])
for h, q0, p0 in [(0.1, 0.9, 0.3), (0.2, 0.9, 0.3), (0.1, 1.1, 0.0), (0.1, 0.5, 1.0)]:
    z0 = np.array([q0, p0, q0**2, q0**4, q0**3]); Z = run(lambda z: kahan_step(Fo, jac(Fo, 5), z, h), z0, 400000)
    q, p, y1, y2, u = Z.T; n = len(q)//10
    worst = [max(abs(c[:n]).max(), 1e-300) for c in (u - q*y1, y2 - y1**2)]; late = [abs(c[-n:]).max() for c in (u - q*y1, y2 - y1**2)]
    dH = p**2/2 + q**8/8 - (p0**2/2 + q0**8/8)
    print(f"octic h={h}, (q0,p0)=({q0},{p0}): u-qy1 first/last {worst[0]:.1e}/{late[0]:.1e}, y2-y1^2 first/last {worst[1]:.1e}/{late[1]:.1e},"
          f" |H-H0| first/last {abs(dH[:n]).max():.1e}/{abs(dH[-n:]).max():.1e}")
# ---- phi^6 Klein-Gordon lattice, hierarchical pointwise lift  w = u^2 (protected), s = u^3 (= u w, triangular)
# u' = v, v' = D2 u - u - s w, w' = 2 u v, s' = 3 w v      (s - u w driven by w - u^2)
N, L = 128, 20.0; dx = 2*L/N; x = -L + dx*np.arange(N)
e = np.ones(N); D2 = sps.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1], format='lil'); D2[0, N-1] = 1; D2[N-1, 0] = 1; D2 = (D2/dx**2).tocsr()
I = sps.identity(N, format='csr'); Z0 = sps.csr_matrix((N, N)); Dg = sps.diags
def F(Z):
    u, v, w, s = Z.reshape(4, N)
    return np.concatenate([v, D2@u - u - s*w, 2*u*v, 3*w*v])
def J(Z):
    u, v, w, s = Z.reshape(4, N)
    return sps.bmat([[Z0, I, Z0, Z0], [D2 - I, Z0, Dg(-s), Dg(-w)], [Dg(2*v), Dg(2*u), Z0, Z0], [Z0, Dg(3*w), Dg(3*v), Z0]], format='csc')
def kahan(Z, h): return Z + h*spla.spsolve((sps.identity(4*N) - 0.5*h*J(Z)).tocsc(), F(Z))
H = lambda u, v: dx*np.sum(0.5*v**2 - 0.5*u*(D2@u) + 0.5*u**2 + u**6/6)
for h in [0.5*dx, 2*dx]:
    u0 = 1.3/np.cosh(x); v0 = np.zeros(N); Z = np.concatenate([u0, v0, u0**2, u0**3]); H0 = H(u0, v0)
    nsteps = int(400/h); rec = []
    for k in range(nsteps):
        Z = kahan(Z, h)
        if k % 50 == 0:
            u, v, w, s = Z.reshape(4, N); rec.append((np.abs(w - u**2).max(), np.abs(s - u*w).max(), abs(H(u, v) - H0)/H0))
    rec = np.array(rec); m = len(rec)//10
    print(f"phi^6 KG h={h/dx:.1f}dx, T=400: max|w-u^2| first/last {rec[:m,0].max():.1e}/{rec[-m:,0].max():.1e};"
          f" max|s-uw| first/last {rec[:m,1].max():.1e}/{rec[-m:,1].max():.1e}; rel. energy err first/last {rec[:m,2].max():.1e}/{rec[-m:,2].max():.1e}")
