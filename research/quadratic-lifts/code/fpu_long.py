import numpy as np, time
from fpu_fast import KL, SV, H, Dm, q0, p0
E0 = H(q0, p0); out = {"E0": E0}
for h in [0.2, 0.5]:
    T = 1e5; n = int(T/h); rec = 50
    for m in ["KL", "SV"]:
        q, p = q0.copy(), p0.copy(); r = Dm(q)**2; ts, es, cs = [], [], []; t0 = time.time()
        for i in range(n):
            if m == "KL": q, p, r = KL(q, p, r, h)
            else: q, p = SV(q, p, h)
            if i % rec == 0:
                ts.append((i+1)*h); es.append(abs(H(q, p) - E0)/E0)
                if m == "KL": cs.append(np.max(abs(r - Dm(q)**2)))
        es = np.array(es); w = len(es)//10
        print(f"h={h} {m}: rel energy error max over windows [0-10%,45-55%,90-100%]: "
              f"{es[:w].max():.2e} {es[len(es)//2-w//2:len(es)//2+w//2].max():.2e} {es[-w:].max():.2e}  (wall {time.time()-t0:.0f}s)"
              + (f"; max|r-d^2| {max(cs):.1e}" if cs else ""))
        out[f"{m}_{h}_t"] = np.array(ts); out[f"{m}_{h}_e"] = es
np.savez("fpu_long.npz", **out)
