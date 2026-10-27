"""Sine-Gordon, lift B: z = (u, v, s, c), standard central differences for u_xx:
   u' = v,  v' = D2 u - s,  s' = c v,  c' = -s v
Energy  E~ = sum v^2/2 - u.D2 u/2 + (1 - c):  Hessian acts on (u, v) only, and u', v' are AFFINE
-> lemma: Kahan preserves  E~ - (h^2/8) [ |v'|^2 + v'.(-D2) v' ...]  i.e. E~_h = E~ - h^2/8 ( |D2 u - s|^2 + v.(-D2) v ) exactly.
Circle constraints: two-variable, A = v_j  ->  (s^2+c^2)/(1+h^2 v^2/4) exact per site.
Risk: the angle u and the phase of (s,c) are tied by a NON-algebraic relation -> may drift."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
N, L = 400, 40.0; dx = 2*L/N; x = -L + dx*np.arange(N)
vel = 0.5; g = 1/np.sqrt(1 - vel**2)
u0 = 4*np.arctan(np.exp(g*(x + 10))) - 4*np.arctan(np.exp(g*(x - 10)))
v0 = -4*g*vel*np.exp(g*(x+10))/(1 + np.exp(2*g*(x+10))) - 4*g*vel*np.exp(g*(x-10))/(1 + np.exp(2*g*(x-10)))
e = np.ones(N); D2 = sps.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1], format='lil'); D2[0, N-1] = 1; D2[N-1, 0] = 1
D2 = (D2/dx**2).tocsr()
Etrue = lambda u, v: dx*np.sum(0.5*v**2 - 0.5*u*(D2@u) + 1 - np.cos(u))
Elift = lambda u, v, s, c: dx*np.sum(0.5*v**2 - 0.5*u*(D2@u) + 1 - c)
Emod = lambda u, v, s, c, h: Elift(u, v, s, c) - h**2/8*dx*np.sum((D2@u - s)**2 - v*(D2@v))
def kahan(u, v, s, c, h):
    Z = sps.csr_matrix((N, N)); I = sps.identity(N, format='csr'); D = sps.diags
    J = sps.bmat([[Z, I, Z, Z], [D2, Z, -I, Z], [Z, D(c), Z, D(v)], [Z, D(-s), D(-v), Z]], format='csc')
    f = np.concatenate([v, D2@u - s, c*v, -s*v])
    d = spla.spsolve((sps.identity(4*N) - 0.5*h*J).tocsc(), h*f)
    return u + d[:N], v + d[N:2*N], s + d[2*N:3*N], c + d[3*N:]
for ratio in [0.5, 2.0]:
    h = ratio*dx; n = int(round(200/h))
    u, v, s, c = u0.copy(), v0.copy(), np.sin(u0), np.cos(u0)
    Em0, Et0 = Emod(u, v, s, c, h), Etrue(u, v); dEm = 0; dEt = []; ph = []
    for k in range(n):
        u, v, s, c = kahan(u, v, s, c, h)
        dEm = max(dEm, abs(Emod(u, v, s, c, h) - Em0))
        if k % 20 == 0:
            dEt.append(abs(Etrue(u, v) - Et0)); ph.append(np.max(abs(np.angle(np.exp(1j*(np.arctan2(s, c) - u))))))
    dEt = np.array(dEt); ph = np.array(ph); m = len(dEt)//10
    print(f"h={ratio}dx: modified-energy drift {dEm/Et0:.1e} (rel); true-energy rel. error first/last 10%: "
          f"{dEt[:m].max()/Et0:.2e} / {dEt[-m:].max()/Et0:.2e};  phase mismatch first/last 10%: {ph[:m].max():.2e} / {ph[-m:].max():.2e}")
