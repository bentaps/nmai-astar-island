import numpy as np, scipy.sparse as sps, time
from kg_pde import setup, energy, KL_step, MQAV_step, STAG_step, SV_step
N, L = 512, 20.0
x, dx, D2 = setup(N, L); I = sps.identity(N, format='csc')
u0 = 2.0/np.cosh(x); v0 = np.zeros(N); E0 = energy(u0, v0, dx, D2)
print("H0 =", E0, " dx =", dx)

def reference(T, h):
    f = lambda u, v: (v, D2@u - u - u**3)
    u, v = u0.copy(), v0.copy()
    for _ in range(int(round(T/h))):
        k1 = f(u, v); k2 = f(u + h/2*k1[0], v + h/2*k1[1]); k3 = f(u + h/2*k2[0], v + h/2*k2[1]); k4 = f(u + h*k3[0], v + h*k3[1])
        u = u + h/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]); v = v + h/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    return u, v

def integrate(m, h, T, every):
    u, v = u0.copy(), v0.copy(); n = int(round(T/h)); ts, dE, mc = [0.0], [0.0], [0.0]; solves = 0
    if m == 'KL': w = u**2
    if m == 'STAG': uh = u + h/2*v + h**2/8*(D2@u - u - u**3); wh = uh**2
    t0 = time.time()
    for k in range(n):
        if m == 'KL': u, v, w = KL_step(u, v, w, h, D2, I); solves += 1
        elif m == 'MQAV': u, v, it = MQAV_step(u, v, h, D2, I); solves += it
        elif m == 'STAG': u, v, wh = STAG_step(u, v, wh, h, D2, I); solves += 1
        else: u, v = SV_step(u, v, h, D2)
        if not np.all(np.isfinite(u)) or np.max(abs(u)) > 1e6: return None
        if (k+1) % every == 0:
            ts.append((k+1)*h); dE.append(energy(u, v, dx, D2) - E0)
            mc.append(np.max(abs(w - u**2)) if m == 'KL' else np.nan)
    return dict(t=np.array(ts), dE=np.array(dE), cons=np.array(mc), u=u, v=v, solves=solves/n, wall=time.time()-t0)

# accuracy at T = 10
T = 10.0
uref, vref = reference(T, dx/8)
print("\nAccuracy at T=10 (max-norm error in u):")
print(" h/dx    KL        MQAV      STAG      SV")
for ratio in [4, 2, 1, 0.5, 0.25]:
    h = ratio*dx; row = []
    for m in ['KL', 'MQAV', 'STAG', 'SV']:
        r = integrate(m, h, T, 10**9)
        row.append(np.nan if r is None else np.max(abs(r['u'] - uref)))
    print(f"{ratio:<6} " + " ".join(f"{e:.2e}" for e in row))

# long time series
series = {}
for ratio, Tl in [(0.5, 1000.0), (4.0, 1000.0)]:
    h = ratio*dx; every = max(1, int(round(1.0/h)))
    for m in ['KL', 'MQAV', 'STAG', 'SV']:
        r = integrate(m, h, Tl, every)
        key = f"{m}_{ratio}"
        if r is None: print(f"{key}: unstable"); continue
        series[key] = r
        print(f"{key}: max|H-H0|/H0 = {np.max(abs(r['dE']))/E0:.2e}, solves/step {r['solves']:.2f}, wall {r['wall']:.1f}s"
              + (f", max|w-u^2| {np.nanmax(r['cons']):.2e}" if m == 'KL' else ""))
np.savez("kg_series.npz", E0=E0, dx=dx, **{f"{k}_t": v['t'] for k, v in series.items()},
         **{f"{k}_dE": v['dE'] for k, v in series.items()}, **{f"{k}_cons": v['cons'] for k, v in series.items()})
