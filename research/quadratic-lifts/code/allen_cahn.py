"""Allen-Cahn u_t = eps^2 u_xx + u - u^3 (periodic), lifted q = u^2 (pointwise), QUADRATIC RHS:
   u_t = eps^2 D2 u + u - u q
   q_t = 2 u eps^2 D2 u + 2 q - 2 q^2 - gamma (q - u^2)
Constraint error c = q - u^2 obeys c_t = (2(1-q) - gamma) c  -> repelling for gamma=0 where |u|<1, attracting for gamma>2.
Integrate with Kahan (linearly implicit). Compare with a standard linearly-implicit IEQ scheme
   (u1-u0)/k = eps^2 D2 ub + ub - u* qb ,  (q1-q0) = 2 u* (u1-u0)    [u* = extrapolation, qb, ub midpoints]
which dissipates the modified energy but lets q drift."""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla
N = 256; Lx = 2*np.pi; dx = Lx/N; x = dx*np.arange(N); eps = 0.08
e = np.ones(N); D2 = sps.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1], format='lil'); D2[0, N-1] = 1; D2[N-1, 0] = 1
D2 = (D2/dx**2).tocsc(); I = sps.identity(N, format='csc')
E = lambda u: dx*np.sum(-0.5*eps**2*u*(D2@u) + 0.25*(u**2 - 1)**2)
rng = np.random.default_rng(0)
u0 = 0.5*np.sin(x) + 0.3*np.cos(3*x) + 0.1*rng.standard_normal(N)

def kahan_ac(u, q, k, gamma):
    # z = (u, q); F_u = eps^2 D2 u + u - u q ; F_q = 2 eps^2 u (D2 u) + 2 q - 2 q^2 - gamma (q - u^2)
    Fu = eps**2*(D2@u) + u - u*q
    D2u = D2@u
    Fq = 2*eps**2*u*D2u + 2*q - 2*q**2 - gamma*(q - u**2)
    Juu = eps**2*D2 + I - sps.diags(q); Juq = -sps.diags(u)
    Jqu = 2*eps**2*(sps.diags(D2u) + sps.diags(u)@D2) + sps.diags(2*gamma*u)
    Jqq = sps.diags(2 - 4*q - gamma)
    J = sps.bmat([[Juu, Juq], [Jqu, Jqq]], format='csc')
    d = spla.spsolve((sps.identity(2*N) - 0.5*k*J).tocsc(), k*np.concatenate([Fu, Fq]))
    return u + d[:N], q + d[N:]

def ieq(u, q, uold, k):
    us = 1.5*u - 0.5*uold
    # q = u^2 - 1:  (u1-u)/k = eps^2 D2 (u+u1)/2 - us*(q + q1)/2,  q1 = q + 2 us (u1 - u)
    # -> (u1-u)/k = eps^2 D2 ub - us*q - us^2 (u1 - u)
    A = I/k - 0.5*eps**2*D2 + sps.diags(us**2)
    rhs = u/k + 0.5*eps**2*(D2@u) - us*q + us**2*u
    u1 = spla.spsolve(A.tocsc(), rhs); q1 = q + 2*us*(u1 - u)
    return u1, q1

# reference: Kahan-free fine explicit Euler is too stiff; use tiny-step IEQ-free semi-implicit (backward Euler on linear part)
def reference(T, k):
    u = u0.copy(); A = (I - k*eps**2*D2).tocsc(); lu = spla.splu(A)
    for _ in range(int(round(T/k))): u = lu.solve(u + k*(u - u**3))
    return u
T = 8.0
u1 = reference(T, 1e-4); u2 = reference(T, 5e-5); uref = 2*u2 - u1   # Richardson -> 2nd order
print(f'reference self-check |u(5e-5)-u(1e-4)| = {np.max(abs(u2-u1)):.1e} (Richardson removes the O(k) part)')
print(f"E(u0) = {E(u0):.5f}, E(ref,T) = {E(uref):.5f}")
for k in [0.01, 0.05, 0.2]:
    n = int(round(T/k))
    for label, method, gamma in [("Kahan, gamma=0  ", 'kahan', 0.0), ("Kahan, gamma=4  ", 'kahan', 4.0), ("IEQ (lin. impl.)", 'ieq', None)]:
        u, q = u0.copy(), u0**2; uold = u0.copy(); maxc = 0; Es = [E(u)]; ok = True
        if method == 'ieq': q = u0**2 - 1
        for _ in range(n):
            if method == 'kahan': u, q = kahan_ac(u, q, k, gamma)
            else: un, q = ieq(u, q, uold, k); uold = u; u = un
            if not np.all(np.isfinite(u)) or abs(u).max() > 1e3: ok = False; break
            c = (q - u**2) if method == 'kahan' else (q - (u**2 - 1))
            maxc = max(maxc, abs(c).max()); Es.append(E(u))
        mono = np.all(np.diff(Es) <= 1e-12) if ok else False
        err = np.max(abs(u - uref)) if ok else np.nan
        if method == 'ieq' and ok:
            Em = dx*np.sum(-0.5*eps**2*u*(D2@u) + 0.25*q**2)
            print(f"      (IEQ modified energy at T = {Em:.5f}, true energy E(u(T)) = {E(u):.5f})")
        print(f"k={k:<5} {label}: ok={ok!s:5s} max|constraint err|={maxc:.2e}  final |c|={abs(c).max() if ok else np.nan:.1e}  "
              f"err vs ref={err:.2e}  energy monotone={mono}")
