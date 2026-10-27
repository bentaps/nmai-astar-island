"""Sine-Gordon u_tt = u_xx - sin u, 'cosine-coupled' lattice in the trig lift (s_j, c_j) = (sin u_j, cos u_j):
   H = sum v^2/2 + (1/dx^2) sum (1 - cos(u_{j+1}-u_j)) + sum (1 - cos u_j)
     = sum v^2/2 + (1/dx^2) sum (1 - c_{j+1} c_j - s_{j+1} s_j) + sum (1 - c_j)     (QUADRATIC in (s,c,v))
   s' = c v,  c' = -s v,  v' = (1/dx^2)[(s_{j+1} c_j - c_{j+1} s_j) - (s_j c_{j-1} - c_j s_{j-1})] - s
Per-site circle constraints s_j^2 + c_j^2 are two-variable with affine factor A = v_j  ->  Kahan preserves
   (s_j^2 + c_j^2)/(1 + h^2 v_j^2/4) exactly at every site.  (No redundant angle variable, so no phase drift.)
Initial data: kink-antikink pair (moving kinks)."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
N, L = 400, 40.0; dx = 2*L/N; x = -L + dx*np.arange(N)
vel = 0.5; g = 1/np.sqrt(1 - vel**2)
u0 = 4*np.arctan(np.exp(g*(x + 10))) - 4*np.arctan(np.exp(g*(x - 10)))      # kink at -10, antikink at +10 (u -> 0 at both ends)
v0 = -4*g*vel*np.exp(g*(x+10))/(1 + np.exp(2*g*(x+10))) * 1.0 - 4*g*vel*np.exp(g*(x-10))/(1 + np.exp(2*g*(x-10)))
s, c, v = np.sin(u0), np.cos(u0), v0.copy()
sh = lambda a, k: np.roll(a, -k)     # sh(a,1)_j = a_{j+1}
def F(s, c, v):
    fv = ((sh(s,1)*c - sh(c,1)*s) - (s*sh(c,-1) - c*sh(s,-1)))/dx**2 - s
    return c*v, -s*v, fv
def H(s, c, v):
    return dx*np.sum(0.5*v**2 + (1 - sh(c,1)*c - sh(s,1)*s)/dx**2 + (1 - c))
def jac(s, c, v):
    Id = sps.identity(N); P = sps.diags([np.ones(N-1)], [1], shape=(N, N), format='lil'); P[N-1, 0] = 1; P = P.tocsr()  # (P a)_j = a_{j+1}
    M = P.T.tocsr()  # (M a)_j = a_{j-1}
    D = sps.diags
    ds_ds = D(np.zeros(N)); ds_dc = D(v); ds_dv = D(c)
    dc_ds = D(-v); dc_dc = D(np.zeros(N)); dc_dv = D(-s)
    # fv_j = (s_{j+1}c_j - c_{j+1}s_j - s_j c_{j-1} + c_j s_{j-1})/dx^2 - s_j
    dv_ds = (D(c)@P - D(sh(c,1)) - D(sh(c,-1)) + D(c)@M)/dx**2 - Id
    dv_dc = (D(sh(s,1)) - D(s)@P - D(s)@M + D(sh(s,-1)))/dx**2
    dv_dv = D(np.zeros(N))
    return sps.bmat([[ds_ds, ds_dc, ds_dv], [dc_ds, dc_dc, dc_dv], [dv_ds, dv_dc, dv_dv]], format='csc')
def kahan(s, c, v, h):
    f = np.concatenate(F(s, c, v)); J = jac(s, c, v)
    d = spla.spsolve((sps.identity(3*N) - 0.5*h*J).tocsc(), h*f)
    return s + d[:N], c + d[N:2*N], v + d[2*N:]
for ratio in [0.5, 2.0]:
    h = ratio*dx; n = int(round(200/h))
    S, C, V = s.copy(), c.copy(), v.copy(); H0 = H(S, C, V)
    K0 = (S**2 + C**2)/(1 + h**2*V**2/4); maxK = 0; maxr = 0; dH = []
    for k in range(n):
        S, C, V = kahan(S, C, V, h)
        maxK = max(maxK, np.max(abs((S**2 + C**2)/(1 + h**2*V**2/4) - K0)))
        maxr = max(maxr, np.max(abs(S**2 + C**2 - 1)))
        if k % 20 == 0: dH.append(H(S, C, V) - H0)
    dH = np.abs(dH)
    print(f"h={ratio}dx: modified circle constraints drift {maxK:.1e}; max|s^2+c^2-1| {maxr:.2e}; "
          f"rel. energy error first/last 10%: {dH[:len(dH)//10].max()/H0:.2e} / {dH[-len(dH)//10:].max()/H0:.2e}")
