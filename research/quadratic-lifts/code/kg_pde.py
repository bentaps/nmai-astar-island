"""phi^4 Klein-Gordon  u_tt = u_xx - u - u^3  on periodic [-L, L), central differences.
Lifted pointwise: w_j = u_j^2.   z = (u, v, w):  u' = v,  v' = D2 u - u - u*w,  w' = 2 u v   (quadratic)
Methods (all 2nd order):
  KL    Kahan on lifted system, Schur-reduced to ONE N x N sparse (cyclic tridiagonal) solve per step
  MQAV  midpoint on lifted system with w eliminated (paper's method)  -> Newton, k solves per step
  STAG  Besse-type staggered: w on half steps, 1 sparse solve per step, exact discrete energy
  SV    Stormer-Verlet (explicit, CFL h < dx)
"""
import numpy as np, scipy.sparse as sps, scipy.sparse.linalg as spla, time

def setup(N=512, L=20.0):
    dx = 2*L/N; x = -L + dx*np.arange(N)
    e = np.ones(N)
    D2 = sps.diags([e[:-1], -2*e, e[:-1]], [-1, 0, 1], format='lil')
    D2[0, N-1] = 1; D2[N-1, 0] = 1
    D2 = (D2/dx**2).tocsc()
    return x, dx, D2

def energy(u, v, dx, D2):
    return dx*np.sum(0.5*v**2 - 0.5*u*(D2@u) + 0.5*u**2 + 0.25*u**4)

def KL_step(u, v, w, h, D2, I):
    Lw = D2 - I - sps.diags(w)
    M = I - (h**2/4)*Lw + sps.diags(h**2/2*u**2 + h**3/4*u*v)
    rhs = h*(D2@u - u - u*w) + (h**2/2)*(Lw@v) - (h/2)*u*(2*h*u*v + h**2*v**2)
    dv = spla.spsolve(M.tocsc(), rhs)
    du = h*v + h/2*dv
    dw = 2*h*u*v + h**2*v**2 + (h**2/2)*v*dv + h*u*dv
    return u + du, v + dv, w + dw

def MQAV_step(u, v, h, D2, I, tol=1e-12, maxit=30):
    un = u + h*v
    for it in range(maxit):
        ub = 0.5*(u + un)
        G = (2/h**2)*(un - u - h*v) - D2@ub + ub + ub*(u**2 + un**2)/2
        J = (2/h**2)*I - 0.5*D2 + 0.5*I + sps.diags(0.5*(u**2 + un**2)/2 + ub*un)
        d = spla.spsolve(J.tocsc(), -G); un = un + d
        if np.max(abs(d)) < tol*(1 + np.max(abs(un))): break
    vn = 2*(un - u)/h - v
    return un, vn, it + 1

def STAG_step(u, v, wh, h, D2, I):
    # (u'-u)/h = (v+v')/2 ;  (v'-v)/h = D2 ub - ub - ub*wh   (wh = w_{n+1/2} frozen)
    A = D2 - I - sps.diags(wh)
    M = (2/h**2)*I - 0.5*A
    rhs = (2/h**2)*(u + h*v) + 0.5*(A@u)
    un = spla.spsolve(M.tocsc(), rhs)
    vn = 2*(un - u)/h - v
    whn = 2*un**2 - wh
    return un, vn, whn

def SV_step(u, v, h, D2):
    f = lambda u: D2@u - u - u**3
    v = v + h/2*f(u); u = u + h*v; v = v + h/2*f(u)
    return u, v

def run_all(h, T, N=512, L=20.0, methods=('KL', 'MQAV', 'STAG', 'SV'), every=10):
    x, dx, D2 = setup(N, L); I = sps.identity(N, format='csc')
    u0 = 2.0/np.cosh(x); v0 = np.zeros(N); E0 = energy(u0, v0, dx, D2)
    n = int(round(T/h)); out = {}
    for m in methods:
        u, v = u0.copy(), v0.copy(); t0 = time.time(); Es, Cs, ts = [], [], []; solves = 0; ok = True
        if m == 'KL': w = u**2
        if m == 'STAG':
            uh = u + h/2*v + h**2/8*(D2@u - u - u**3); wh = uh**2
            Ed = []
        for k in range(n):
            if m == 'KL': u, v, w = KL_step(u, v, w, h, D2, I); solves += 1
            elif m == 'MQAV': u, v, it = MQAV_step(u, v, h, D2, I); solves += it
            elif m == 'STAG':
                whold = wh; u, v, wh = STAG_step(u, v, wh, h, D2, I); solves += 1
                Ed.append(dx*np.sum(0.5*v**2 - 0.5*u*(D2@u) + 0.5*u**2 + 0.25*whold*wh))
            else: u, v = SV_step(u, v, h, D2)
            if not np.all(np.isfinite(u)) or np.max(abs(u)) > 1e6: ok = False; break
            if (k+1) % every == 0:
                ts.append((k+1)*h); Es.append(energy(u, v, dx, D2) - E0)
                if m == 'KL': Cs.append(np.max(abs(w - u**2 + h**2/4*v**2 - (h**2/4)*v0**2)))
        res = dict(t=np.array(ts), dE=np.array(Es), ok=ok, time=time.time()-t0, solves=solves/max(k+1,1), u=u, v=v)
        if m == 'KL': res['modcons'] = np.array(Cs); res['cons'] = np.max(abs(w - u**2))
        if m == 'STAG': res['Ed_drift'] = np.max(abs(np.array(Ed) - Ed[0])) if Ed else np.nan
        out[m] = res
    return out, dx

if __name__ == "__main__":
    import sys
    for ratio, T in [(0.5, 200.0), (4.0, 200.0)]:
        x, dx, D2 = setup(); h = ratio*dx
        out, dx = run_all(h, T, methods=('KL', 'MQAV', 'STAG', 'SV'), every=max(1, int(0.5/h)))
        print(f"\n=== h = {ratio} dx = {h:.4f},  T = {T} ===")
        for m, r in out.items():
            extra = ""
            if m == 'KL': extra = f" | modified-constraint drift {np.max(r['modcons']):.1e}, max|w-u^2| at end {r['cons']:.1e}"
            if m == 'STAG': extra = f" | discrete-energy drift {r['Ed_drift']:.1e}"
            print(f"{m:5s} ok={r['ok']!s:5s} max|H-H0|={np.max(abs(r['dE'])) if len(r['dE']) else np.nan:.2e}  "
                  f"solves/step={r['solves']:.2f}  wall={r['time']:.1f}s" + extra)
