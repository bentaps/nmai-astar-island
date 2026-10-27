"""FPU-beta chain, fixed ends: H = sum p^2/2 + sum_{j=0}^{N} [ r_j/2 + beta r_j^2/4 ],  r_j = (q_{j+1}-q_j)^2, q_0=q_{N+1}=0.
Lift r_j = d_j^2 with d = Dq (differences: affine dynamics d' = Dp)  -> position-only lift:
   q' = p,  p' = -D^T( d + beta d*r ),  r' = 2 d*(D p)          (quadratic; constraint factors affine)
Kahan: one sparse solve/step; preserves r_j - d_j^2 + (h^2/4)(D p)_j^2 exactly.  Compare SV and MQAV midpoint."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time
N, beta = 32, 1.0
D = sps.lil_matrix((N+1, N))
for j in range(N+1):
    if j < N: D[j, j] = 1
    if j > 0: D[j, j-1] = -1
D = D.tocsr()                       # (Dq)_j = q_j - q_{j-1} with q_0 = q_{N+1} = 0 (indices shifted)
H = lambda q, p: 0.5*p@p + np.sum(0.5*(D@q)**2 + beta/4*(D@q)**4)
# initial condition: energy in the lowest normal mode
k = np.arange(1, N+1)
A = 4.0
q0 = A*np.sin(np.pi*k/(N+1)); p0 = np.zeros(N)
E0 = H(q0, p0)
def SV(q, p, h):
    f = lambda q: -D.T@(D@q + beta*(D@q)**3)
    p = p + h/2*f(q); q = q + h*p; p = p + h/2*f(q); return q, p
def KL(q, p, r, h):
    d = D@q; Dp = D@p
    # z = (q, p, r);  F = (p, -D^T(d + beta d r), 2 d Dp)
    Fq, Fp, Fr = p, -D.T@(d + beta*d*r), 2*d*Dp
    I = sps.identity(N); Z = sps.csr_matrix((N, N)); Zr = sps.csr_matrix((N, N+1))
    dFp_dq = -D.T@sps.diags(1 + beta*r)@D; dFp_dr = -D.T@sps.diags(beta*d)
    dFr_dq = sps.diags(2*Dp)@D; dFr_dp = sps.diags(2*d)@D
    J = sps.bmat([[Z, I, Zr], [dFp_dq, Z, dFp_dr], [dFr_dq, dFr_dp, sps.csr_matrix((N+1, N+1))]], format='csc')
    dz = spla.spsolve((sps.identity(3*N+1) - 0.5*h*J).tocsc(), h*np.concatenate([Fq, Fp, Fr]))
    return q + dz[:N], p + dz[N:2*N], r + dz[2*N:]
def MQAV(q, p, h, tol=1e-13):
    # (q'-q)/h = pb ; (p'-p)/h = -D^T( db + beta db * (d^2 + d'^2)/2 ),  Newton in q'
    qn = q + h*p; d = D@q
    for it in range(30):
        dn = D@qn; db = 0.5*(d + dn)
        G = (2/h**2)*(qn - q - h*p) + D.T@(db + beta*db*(d**2 + dn**2)/2)
        Jm = (2/h**2)*sps.identity(N) + D.T@sps.diags(0.5 + beta*(0.5*(d**2 + dn**2)/2 + db*dn))@D
        dq = spla.spsolve(Jm.tocsc(), -G); qn = qn + dq
        if np.max(abs(dq)) < tol*(1 + np.max(abs(qn))): break
    return qn, 2*(qn - q)/h - p, it + 1
if __name__ == "__main__":
    T = 2e4
    for h in [0.2, 0.5]:
        n = int(T/h); rec = max(1, n//400)
        res = {}
        for m in ['SV', 'KL', 'MQAV']:
            q, p = q0.copy(), p0.copy(); r = (D@q)**2; errs = []; its = 0; t0 = time.time(); ok = True
            for i in range(n):
                if m == 'SV': q, p = SV(q, p, h)
                elif m == 'KL': q, p, r = KL(q, p, r, h)
                else: q, p, it = MQAV(q, p, h); its += it
                if not np.all(np.isfinite(q)): ok = False; break
                if i % rec == 0: errs.append(abs(H(q, p) - E0)/E0)
            errs = np.array(errs); m10 = max(1, len(errs)//10)
            extra = f", Newton its/step {its/n:.2f}" if m == 'MQAV' else ""
            if m == 'KL': extra = f", max|r-d^2| at end {np.max(abs(r-(D@q)**2)):.1e}"
            print(f"h={h} {m:5s} ok={ok}: rel. energy error first/last 10%: {errs[:m10].max():.2e} / {errs[-m10:].max():.2e}, wall {time.time()-t0:.0f}s{extra}")
