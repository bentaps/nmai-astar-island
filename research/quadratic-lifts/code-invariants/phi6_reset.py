import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
N, L = 128, 20.0; dx = 2*L/N; x = -L + dx*np.arange(N)
e = np.ones(N); D2 = sps.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1], format='lil'); D2[0, N-1] = 1; D2[N-1, 0] = 1; D2 = (D2/dx**2).tocsr()
I = sps.identity(N, format='csr'); Z0 = sps.csr_matrix((N, N)); Dg = sps.diags
def F(Z):
    u, v, w, s = Z.reshape(4, N); return np.concatenate([v, D2@u - u - s*w, 2*u*v, 3*w*v])
def J(Z):
    u, v, w, s = Z.reshape(4, N)
    return sps.bmat([[Z0, I, Z0, Z0], [D2 - I, Z0, Dg(-s), Dg(-w)], [Dg(2*v), Dg(2*u), Z0, Z0], [Z0, Dg(3*w), Dg(3*v), Z0]], format='csc')
kahan = lambda Z, h: Z + h*spla.spsolve((sps.identity(4*N) - 0.5*h*J(Z)).tocsc(), F(Z))
H = lambda u, v: dx*np.sum(0.5*v**2 - 0.5*u*(D2@u) + 0.5*u**2 + u**6/6)
for h in [0.5*dx, 2*dx]:
    for mode in ["no reset", "reset s := u w"]:
        u0 = 1.3/np.cosh(x); v0 = np.zeros(N); Z = np.concatenate([u0, v0, u0**2, u0**3]); H0 = H(u0, v0); rec = []
        for k in range(int(400/h)):
            Z = kahan(Z, h)
            if mode != "no reset":
                u, v, w, s = Z.reshape(4, N); Z[3*N:] = u*w
            if k % 50 == 0:
                u, v, w, s = Z.reshape(4, N); rec.append((np.abs(w - u**2).max(), abs(H(u, v) - H0)/H0))
        rec = np.array(rec); m = len(rec)//10
        print(f"h={h/dx:.1f}dx {mode:15s}: max|w-u^2| first/last {rec[:m,0].max():.1e}/{rec[-m:,0].max():.1e}; rel. energy err first/mid/last "
              f"{rec[:m,1].max():.1e}/{rec[4*m:5*m,1].max():.1e}/{rec[-m:,1].max():.1e}")
