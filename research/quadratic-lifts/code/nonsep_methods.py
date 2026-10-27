"""H = x1^2/2 + x2^4 + x1^2 x2^2 (paper, Sec. 5.1).  x1' = -dH/dx2, x2' = dH/dx1  (x2 ~ q, x1 ~ p).
Methods:
  MQAV : paper's MQAV midpoint with alpha = 0  (Newton)            -> exact H
  STAG : Besse-type staggered QAV scheme, y on half steps (1 linear 2x2 solve) -> exact modified energy
  SPLIT: explicit Lie-Poisson splitting  H~ = y11/2 + y22^2 + y12^2  (each piece exactly solvable)
"""
import numpy as np
H = lambda x1, x2: x1**2/2 + x2**4 + x1**2*x2**2
f = lambda x: np.array([-2*x[0]**2*x[1] - 4*x[1]**3, 2*x[0]*x[1]**2 + x[0]])

def mqav_step(x, h, tol=1e-14):
    # alpha = 0: (x1'-x1)/h = -2 xb1 * yb12 - 4 xb2 * yb22 ; (x2'-x2)/h = 2 xb2 yb12 + xb1
    def G(xn):
        xb = (x + xn)/2; yb12 = (x[0]*x[1] + xn[0]*xn[1])/2; yb22 = (x[1]**2 + xn[1]**2)/2
        return xn - x - h*np.array([-2*xb[0]*yb12 - 4*xb[1]*yb22, 2*xb[1]*yb12 + xb[0]])
    xn = x + h*f(x)
    for it in range(60):
        g = G(xn)
        J = np.empty((2, 2)); e = 1e-7
        for k in range(2):
            d = np.zeros(2); d[k] = e; J[:, k] = (G(xn + d) - G(xn - d))/(2*e)
        dx = np.linalg.solve(J, -g); xn = xn + dx
        if np.max(abs(dx)) < tol*(1 + np.max(abs(xn))): return xn, it+1, True
        if not np.all(np.isfinite(xn)) or np.max(abs(xn)) > 1e6: return xn, it+1, False
    return xn, 60, False

def stag_run(x0, h, n):
    # H~ = x1^2/2 + y22^2 + y12^2 (alpha = 0);  phi(x) = (x1 x2, x2^2) = (y12, y22)
    # (x' - x)/h = J [A xb + Dphi(xb)^T B y_half],  J = [[0,-1],[1,0]] (x1' = -d/dx2, x2' = d/dx1)
    # grad_x: d/dx1 = x1 + 2 y12 * x2 ;  d/dx2 = 2 y12 * x1 + 2*2 y22 * x2   (Dphi^T B y with B = diag(2,2))
    x = np.array(x0, float)
    # y_{1/2} ~ phi(x(h/2)): use one explicit midpoint-ish predictor
    xh = x + h/2*f(x + h/4*f(x))
    y = np.array([xh[0]*xh[1], xh[1]**2])
    E = np.empty(n); Hs = np.empty(n+1); Hs[0] = H(*x)
    for k in range(n):
        y12, y22 = y
        # linear system for x' with xb = (x+x')/2:
        # x1' - x1 = -h [2 y12 xb1 + 4 y22 xb2] ; x2' - x2 = h [xb1 + 2 y12 xb2]
        M = np.array([[1 + h*y12, 2*h*y22], [-h/2, 1 - h*y12]])
        rhs = np.array([x[0] - h*y12*x[0] - 2*h*y22*x[1], x[1] + h/2*x[0] + h*y12*x[1]])
        xn = np.linalg.solve(M, rhs)
        yn = 2*np.array([xn[0]*xn[1], xn[1]**2]) - y
        E[k] = 0.5*xn[0]**2 + y[0]*yn[0] + y[1]*yn[1]     # E^{n+1} = x1^2/2 + y_{n+1/2}.y_{n+3/2} (B=diag(1,1)*... )
        x, y = xn, yn; Hs[k+1] = H(*x)
        if not np.all(np.isfinite(x)) or np.max(abs(x)) > 1e6: return Hs[:k+2], E[:k+1], False
    return Hs, E, True

def split_step(x, h):
    x1, x2 = x
    def flow3(x1, x2, t):   # H3 = (x1 x2)^2: G = x1 x2 const; x2' = 2 G x2, x1' = -2 G x1
        G = x1*x2; return x1*np.exp(-2*G*t), x2*np.exp(2*G*t)
    x1, x2 = flow3(x1, x2, h/2)
    x1 -= h/2*4*x2**3            # H2 = x2^4: x1' = -4 x2^3
    x2 += h*x1                   # H1 = x1^2/2: x2' = x1
    x1 -= h/2*4*x2**3
    x1, x2 = flow3(x1, x2, h/2)
    return np.array([x1, x2])

if __name__ == "__main__":
    h, n = 0.1, 10000
    print("Stability test as in the paper (h=0.1, 1e4 steps, x0=(2+2i/3, 0)):")
    print(" i   x1_0   MQAV(maxdH, its)      STAG(maxdH, discrete-E drift)   SPLIT(maxdH)")
    for i in range(0, 16):
        x0 = np.array([2 + 2*i/3, 0.0]); H0 = H(*x0)
        # MQAV
        x = x0.copy(); ok = True; md = 0; its = 0
        for k in range(n):
            x, it, conv = mqav_step(x, h); its += it
            if not conv: ok = False; break
            md = max(md, abs(H(*x) - H0))
        mq = f"{md:.1e},{its/(k+1):.1f}" if ok else f"FAIL@{k}"
        Hs, E, ok2 = stag_run(x0, h, n)
        st = f"{np.max(abs(Hs-H0)):.1e}, {np.max(abs(E-E[0])):.0e}" if ok2 else f"FAIL@{len(Hs)}"
        x = x0.copy(); md = 0; ok3 = True
        for k in range(n):
            x = split_step(x, h)
            if not np.all(np.isfinite(x)) or abs(x).max() > 1e6: ok3 = False; break
            md = max(md, abs(H(*x) - H0))
        sp = f"{md:.1e}" if ok3 else f"FAIL@{k}"
        print(f"{i:2d}  {x0[0]:5.2f}   {mq:22s} {st:30s} {sp}")
