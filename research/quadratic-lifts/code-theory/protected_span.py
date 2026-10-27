"""Automatic 'Kahan-friendliness' analysis of a quadratic vector field F(z) = b + B z + Q(z,z).

1. space of quadratic first integrals  I(z) = 1/2 z'Sz + s'z   (solve grad I . F == 0)
2. protected members:
   (L)  affine-Hessian / companion lemma:  S Q(u,v) = 0 for all u,v   -> I - h^2/8 F'SF conserved
   (T)  two-variable (CMOQ after a linear change of coords): rank [S | s] <= 2
        (searched over the integral space; complex members contribute Re/Im)
3. protected span P = span(L, T);  report which given constraints / energies lie in P."""
import numpy as np
from itertools import combinations_with_replacement

def coeffs(F, N):
    """b, B, Q (Q[k] symmetric matrices) of a quadratic field, exact up to rounding."""
    I = np.eye(N); b = F(np.zeros(N))
    Q = np.zeros((N, N, N)); B = np.zeros((N, N))
    for i in range(N):
        fp, fm = F(I[i]), F(-I[i])
        B[:, i] = (fp - fm)/2; Q[:, i, i] = (fp + fm)/2 - b
    for i in range(N):
        for j in range(i+1, N):
            fij = F(I[i] + I[j]) - b - B[:, i] - B[:, j] - Q[:, i, i] - Q[:, j, j]
            Q[:, i, j] = Q[:, j, i] = fij/2
    return b, B, Q

def unpack(v, N):
    S = np.zeros((N, N)); idx = 0
    for a in range(N):
        for c in range(a, N):
            S[a, c] = S[c, a] = v[idx]; idx += 1
    return S, v[idx:idx+N]

def integrals(F, N, nsamp=None, tol=1e-9, rng=np.random.default_rng(0)):
    nS = N*(N+1)//2; nunk = nS + N
    nsamp = nsamp or 4*nunk
    rows = []
    for _ in range(nsamp):
        z = rng.standard_normal(N); f = F(z); row = []
        for a in range(N):
            for c in range(a, N):
                row.append(z[c]*f[a] + z[a]*f[c] if a != c else z[a]*f[a])
        row += list(f); rows.append(row)
    A = np.array(rows); A /= np.linalg.norm(A, axis=1, keepdims=True)
    U, s, Vt = np.linalg.svd(A); s = s/s[0]
    k = int(np.sum(s < tol)) + (nunk - len(s) if nunk > len(s) else 0)
    return Vt[nunk-k:]                      # rows: basis of integrals (vector form)

def lemma_subspace(basis, Q, N):
    """members with S Q(e_j,e_k) = 0 for all j,k"""
    if len(basis) == 0: return basis
    cols = []
    for v in basis:
        S, s = unpack(v, N)
        cols.append(np.einsum('ab,bjk->ajk', S, Q).ravel())
    C = np.array(cols).T                    # (N*N*N) x r
    if np.allclose(C, 0): return basis
    U, sv, Vt = np.linalg.svd(C); r = len(basis)
    null = Vt[int(np.sum(sv > 1e-9*max(sv[0], 1))):]
    return null @ basis

def twovar_members(basis, N, rng=np.random.default_rng(1), tol=1e-9):
    """All (real or complex) members lam of the integral space with rank[S(lam)|s(lam)] <= 2.
    r = 1, 2: exact (pencil cubic);  r = 3: exact (intersection of two plane cubics by a resultant);
    r >= 4: pencils between pairs of basis members (heuristic, may miss members)."""
    r = len(basis)
    if r == 0: return np.zeros((0, 0))
    mats = np.array([np.hstack([unpack(v, N)[0], unpack(v, N)[1][:, None]]) for v in basis])
    M_of = lambda c: np.tensordot(c, mats, axes=1)
    def rank_ok(c):
        sv = np.linalg.svd(M_of(c), compute_uv=False)
        return len(sv) <= 2 or sv[2] < tol*sv[0]
    found = []
    def pencil(a, b):
        P = rng.standard_normal((3, N)); R = rng.standard_normal((N+1, 3))
        ts = np.exp(2j*np.pi*np.arange(8)/8)
        vals = [np.linalg.det(P@M_of(a + t*b)@R) for t in ts]
        coef = np.polyfit(ts, vals, 3)
        cands = [a + t*b for t in np.roots(coef)] + [b]
        return [c for c in cands if rank_ok(c)]
    if r == 1:
        if rank_ok(np.ones(1)): found.append(np.ones(1, complex))
    elif r == 2:
        found += pencil(np.eye(2)[0].astype(complex), np.eye(2)[1].astype(complex))
    elif r == 3:
        Pi = rng.standard_normal((3, 3))
        lam = lambda s_, t_: Pi@np.array([1, s_, t_])
        comps = [(rng.standard_normal((3, N)), rng.standard_normal((N+1, 3))) for _ in range(2)]
        def fit(P, R):          # bivariate cubic coefficients by least squares
            mons = [(i, j) for i in range(4) for j in range(4 - i)]
            pts = rng.standard_normal((40, 2)) + 1j*rng.standard_normal((40, 2))
            A = np.array([[p[0]**i*p[1]**j for (i, j) in mons] for p in pts])
            b = np.array([np.linalg.det(P@M_of(lam(*p))@R) for p in pts])
            return dict(zip(mons, np.linalg.lstsq(A, b, rcond=None)[0]))
        f, g = fit(*comps[0]), fit(*comps[1])
        tpoly = lambda F, s_: np.array([sum(F[(i, j)]*s_**i for i in range(4 - j)) for j in range(3, -1, -1)])
        def sylv(a, b):
            Sm = np.zeros((6, 6), complex)
            for k in range(3): Sm[k, k:k+4] = a
            for k in range(3): Sm[3+k, k:k+4] = b
            return np.linalg.det(Sm)
        K = 32; ss = np.exp(2j*np.pi*np.arange(K)/K)*1.3
        vals = np.array([sylv(tpoly(f, s_), tpoly(g, s_)) for s_ in ss])
        coef = np.fft.fft(vals)/K; coef = coef/(1.3**np.arange(K))      # coefficients of s^k
        deg = max(k for k in range(K) if abs(coef[k]) > 1e-10*np.max(abs(coef)))
        for s_ in np.roots(coef[:deg+1][::-1]):
            for t_ in np.roots(tpoly(f, s_)):
                c = lam(s_, t_)
                if rank_ok(c): found.append(c)
    else:
        for i in range(r):
            for j in range(i+1, r):
                found += pencil(np.eye(r)[i].astype(complex), np.eye(r)[j].astype(complex))
    if not found: return np.zeros((0, r))
    Fm = np.array(found); Rspan = np.vstack([Fm.real, Fm.imag])
    U, sv, Vt = np.linalg.svd(Rspan)
    return Vt[:int(np.sum(sv > 1e-6*sv[0]))]

def analyse(name, F, N, named):
    """named: dict label -> (S, s) of quadratic functions to test for protection (constraints, energies)."""
    b, B, Q = coeffs(F, N)
    basis = integrals(F, N)
    r = len(basis)
    L = lemma_subspace(basis, Q, N)
    Tco = twovar_members(basis, N)
    T = Tco @ basis if len(Tco) else np.zeros((0, basis.shape[1]))
    P = np.vstack([L, T]) if (len(L) or len(T)) else np.zeros((0, basis.shape[1]))
    if len(P):
        U, sv, Vt = np.linalg.svd(P); P = Vt[:int(np.sum(sv > 1e-6*sv[0]))]
    def vec(S, s):
        v = []
        for a in range(N):
            for c in range(a, N): v.append(S[a, c])
        return np.array(v + list(s), float)
    status = {}
    for lab, (S, s) in named.items():
        v = vec(np.asarray(S, float), np.asarray(s, float)); v /= np.linalg.norm(v)
        in_int = np.linalg.norm(v - basis.T@np.linalg.lstsq(basis.T, v, rcond=None)[0]) < 1e-6 if r else False
        in_P = (np.linalg.norm(v - P.T@np.linalg.lstsq(P.T, v, rcond=None)[0]) < 1e-6) if len(P) else False
        status[lab] = "protected" if in_P else ("integral, NOT protected" if in_int else "not an exact integral")
    print(f"{name}: dim={N}, quadratic integrals={r}, lemma-protected={len(L)}, two-variable span={len(T)}, protected span={len(P)}")
    for lab, st in status.items(): print(f"     {lab:28s} {st}")
    return status
