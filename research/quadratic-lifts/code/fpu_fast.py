"""FPU-beta, fast O(N) Kahan-lift via Schur reduction to a tridiagonal system in p'.  Records full energy series."""
import numpy as np, time
from scipy.linalg import solve_banded
N, beta, A = 32, 1.0, 4.0
k = np.arange(1, N+1)
q0 = A*np.sin(np.pi*k/(N+1)); p0 = np.zeros(N)
Dm = lambda q: np.diff(np.concatenate([[0.0], q, [0.0]]))          # d_j = q_j - q_{j-1}, length N+1
DT = lambda d: d[:-1] - d[1:]                                         # D^T d, length N
H = lambda q, p: 0.5*p@p + np.sum(0.5*Dm(q)**2 + beta/4*Dm(q)**4)
def DTdiagD_banded(c):
    """banded form of D^T diag(c) D (c length N+1): tridiagonal, ab[0]=super, ab[1]=diag, ab[2]=sub"""
    ab = np.zeros((3, N)); ab[1] = c[:-1] + c[1:]; ab[0, 1:] = -c[1:-1]; ab[2, :-1] = -c[1:-1]
    return ab
def KL(q, p, r, h):
    d, a = Dm(q), Dm(p)
    c = h/2 + beta*(h*d**2 + h**2/2*d*a + h/2*r)
    ab = (h/2)*DTdiagD_banded(c); ab[1] += 1.0
    rhs = p - (h/2)*DT(2*d + h/2*a + beta*(d*(r + h*d*a + h**2/2*a**2) + (d + h/2*a)*r))
    pn = solve_banded((1, 1), ab, rhs)
    b = Dm(pn)
    qn = q + h/2*(p + pn)
    rn = r + h*d*a + h**2/2*a**2 + (h*d + h**2/2*a)*b
    return qn, pn, rn
def SV(q, p, h):
    f = lambda q: -DT(Dm(q) + beta*Dm(q)**3)
    p = p + h/2*f(q); q = q + h*p; p = p + h/2*f(q); return q, p
if __name__ == "__main__":
    # check against the generic sparse implementation for one step
    import fpu as slow
    qq, pp, rr = KL(q0, p0 + 0.3*np.cos(k), Dm(q0)**2, 0.2)
    qs, ps, rs = slow.KL(q0, p0 + 0.3*np.cos(k), (slow.D@q0)**2, 0.2)
    print("fast vs slow max diff:", max(abs(qq-qs).max(), abs(pp-ps).max(), abs(rr-rs).max()))
