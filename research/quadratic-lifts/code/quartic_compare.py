"""Quartic oscillator H = p^2/2 + q^4/4: compare integrators on long runs.
Methods
  KL    : Kahan on the lifted system (q,p,y), y0 = q0^2                   [1 linear 3x3 solve/step]
  KLm   : Kahan on lifted system, modified start y0 = q0^2 - h^2 p0^2/4    [one-step map on (q,p)]
  MQAV  : midpoint on lifted system with y eliminated (= AVF here)       [Newton]
  SV    : Stormer-Verlet                                                 [explicit]
  IEQ   : linearly implicit 'SAV/IEQ-type' scheme on lifted system:
          (z1-z0)/h = K(z*) grad H~((z0+z1)/2), z* = (3 z0 - z_{-1})/2   [1 linear solve/step]
  STAG  : Besse-type staggered scheme: y on half steps,
          (y_{n+1/2}+y_{n-1/2})/2 = q_n^2 ; (q,p) midpoint with y_{n+1/2} frozen [1 linear 2x2 solve/step]
"""
import numpy as np, time
from qlift import kahan_step

H = lambda q, p: 0.5*p**2 + 0.25*q**4
F = lambda z: np.array([z[1], -z[0]*z[2], 2*z[0]*z[1]])
JF = lambda z: np.array([[0, 1, 0], [-z[2], 0, -z[0]], [2*z[1], 2*z[0], 0]])

def KL(q0, p0, h, n, modified=False):
    z = np.array([q0, p0, q0**2 - (h**2/4*p0**2 if modified else 0.0)])
    out = np.empty((n+1, 3)); out[0] = z
    for k in range(n):
        z = kahan_step(F, JF, z, h); out[k+1] = z
    return out[:, 0], out[:, 1], out[:, 2]

def MQAV(q0, p0, h, n, tol=1e-15):
    # (q1-q0)/h = (p0+p1)/2 ; (p1-p0)/h = -(q0+q1)/2 * (q0^2+q1^2)/2
    q, p = q0, p0; Q = np.empty(n+1); P = np.empty(n+1); Q[0], P[0] = q, p; its = 0
    for k in range(n):
        q1, p1 = q + h*p, p - h*q**3
        for _ in range(50):
            its += 1
            g1 = q1 - q - h*(p + p1)/2
            g2 = p1 - p + h*(q + q1)/2*(q**2 + q1**2)/2
            J = np.array([[1, -h/2], [h/4*((q**2+q1**2) + (q+q1)*2*q1), 1]])
            d = np.linalg.solve(J, [-g1, -g2]); q1 += d[0]; p1 += d[1]
            if abs(d).max() < tol*(1+abs(q1)+abs(p1)): break
        q, p = q1, p1; Q[k+1], P[k+1] = q, p
    return Q, P, its/n

def SV(q0, p0, h, n):
    q, p = q0, p0; Q = np.empty(n+1); P = np.empty(n+1); Q[0], P[0] = q, p
    for k in range(n):
        p -= h/2*q**3; q += h*p; p -= h/2*q**3
        Q[k+1], P[k+1] = q, p
    return Q, P

def IEQ(q0, p0, h, n):
    # first step with KL to get z_{1}; then linearly implicit with extrapolated K
    K = lambda q: np.array([[0, 1, 0], [-1, 0, -2*q], [0, 2*q, 0]])
    G = np.diag([0, 1, 0.5])        # grad H~ = G z  (H~ = p^2/2 + y^2/4)
    zm = np.array([q0, p0, q0**2]); z = kahan_step(F, JF, zm, h)
    Z = np.empty((n+1, 3)); Z[0], Z[1] = zm, z
    for k in range(1, n):
        qs = 1.5*z[0] - 0.5*zm[0]
        A = K(qs) @ G
        z1 = np.linalg.solve(np.eye(3) - h/2*A, z + h/2*A@z)
        zm, z = z, z1; Z[k+1] = z
    return Z[:, 0], Z[:, 1], Z[:, 2]

def STAG(q0, p0, h, n):
    # y_{1/2} from a half step of the exact relation (use Taylor: q(h/2)^2 ~ (q0 + h/2 p0)^2 - h^2/4 q0^4/...)
    q, p = q0, p0
    qh = q0 + h/2*p0 - h**2/8*q0**3            # q(h/2) to O(h^3)
    y_half = qh**2                               # y_{1/2} ~ q(h/2)^2
    Q = np.empty(n+1); P = np.empty(n+1); Y = np.empty(n+1); Q[0], P[0] = q, p; Y[0] = y_half
    for k in range(n):
        # (q1-q)/h = (p+p1)/2 ; (p1-p)/h = -(q+q1)/2 * y_half
        a = y_half
        M = np.array([[1, -h/2], [h/2*a, 1]])
        rhs = np.array([q + h/2*p, p - h/2*a*q])
        q1, p1 = np.linalg.solve(M, rhs)
        y_half = 2*q1**2 - y_half                 # (y_{n+3/2} + y_{n+1/2})/2 = q_{n+1}^2
        q, p = q1, p1; Q[k+1], P[k+1] = q, p; Y[k+1] = y_half
    return Q, P, Y

if __name__ == "__main__":
    q0, p0 = 1.0, 0.5; H0 = H(q0, p0)
    h = 0.1; T = 1e4; n = int(T/h)
    res = {}
    t = time.time(); Q, P, Y = KL(q0, p0, h, n); res['KL'] = (Q, P, time.time()-t)
    print(f"KL : max|H-H0|={np.max(abs(H(Q,P)-H0)):.3e}  max|y-q^2|={np.max(abs(Y-Q**2)):.3e}  "
          f"modified-constraint drift={np.max(abs(Y-Q**2+h**2/4*P**2-(Y[0]-Q[0]**2+h**2/4*P[0]**2))):.1e}  "
          f"modified-energy drift={np.max(abs((P**2/2+Y**2/4)/(1+h**2*Q**2/2)-(P[0]**2/2+Y[0]**2/4)/(1+h**2*Q[0]**2/2))):.1e}")
    t = time.time(); Q, P, Y = KL(q0, p0, h, n, modified=True); res['KLm'] = (Q, P, time.time()-t)
    print(f"KLm: max|H-H0|={np.max(abs(H(Q,P)-H0)):.3e}  y-(q^2-h^2p^2/4) max={np.max(abs(Y-Q**2+h**2/4*P**2)):.1e}")
    t = time.time(); Q, P, its = MQAV(q0, p0, h, n); res['MQAV'] = (Q, P, time.time()-t)
    print(f"MQAV: max|H-H0|={np.max(abs(H(Q,P)-H0)):.3e}  avg Newton its/step={its:.2f}")
    t = time.time(); Q, P = SV(q0, p0, h, n); res['SV'] = (Q, P, time.time()-t)
    print(f"SV : max|H-H0|={np.max(abs(H(Q,P)-H0)):.3e}")
    t = time.time(); Q, P, Y = IEQ(q0, p0, h, n); res['IEQ'] = (Q, P, time.time()-t)
    Ht = P**2/2 + Y**2/4
    print(f"IEQ: max|H-H0|={np.max(abs(H(Q,P)-H0)):.3e}  max|H~-H~0|={np.max(abs(Ht-Ht[0])):.1e}  "
          f"|y-q^2| at end={abs(Y[-1]-Q[-1]**2):.3e}, max={np.max(abs(Y-Q**2)):.3e}")
    t = time.time(); Q, P, Y = STAG(q0, p0, h, n); res['STAG'] = (Q, P, time.time()-t)
    E = P[1:]**2/2 + 0.25*Y[:-1]*Y[1:]   # E^n = p_n^2/2 + y_{n-1/2} y_{n+1/2}/4
    print(f"STAG: max|H-H0|={np.max(abs(H(Q,P)-H0)):.3e}  discrete energy drift={np.max(abs(E-E[0])):.1e}")
    np.savez("quartic_long.npz", **{k: np.vstack([v[0], v[1]]) for k, v in res.items()}, h=h)
