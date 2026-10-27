import numpy as np
from quartic_compare import KL, MQAV, SV, STAG, IEQ
# reference solution via very fine MQAV (energy-exact, 2nd order) + Richardson is overkill; use RK4 fine
def rk4(q0, p0, T, n):
    f = lambda z: np.array([z[1], -z[0]**3]); z = np.array([q0, p0]); h = T/n
    for _ in range(n):
        k1 = f(z); k2 = f(z+h/2*k1); k3 = f(z+h/2*k2); k4 = f(z+h*k3); z = z + h/6*(k1+2*k2+2*k3+k4)
    return z
q0, p0, T = 1.0, 0.5, 10.0
ref = rk4(q0, p0, T, 200000)
print(" h        KL        KLm       MQAV      SV        STAG")
prev = None
for h in [0.2, 0.1, 0.05, 0.025, 0.0125]:
    n = int(round(T/h))
    e = []
    for meth in ['KL', 'KLm', 'MQAV', 'SV', 'STAG']:
        if meth == 'KL': Q, P, _ = KL(q0, p0, h, n)
        elif meth == 'KLm': Q, P, _ = KL(q0, p0, h, n, modified=True)
        elif meth == 'MQAV': Q, P, _ = MQAV(q0, p0, h, n)
        elif meth == 'SV': Q, P = SV(q0, p0, h, n)
        else: Q, P, _ = STAG(q0, p0, h, n)
        e.append(np.hypot(Q[-1]-ref[0], P[-1]-ref[1]))
    e = np.array(e)
    rates = "" if prev is None else "  rates: " + " ".join(f"{r:.2f}" for r in np.log2(prev/e))
    print(f"{h:<8} " + " ".join(f"{x:.2e}" for x in e) + rates)
    prev = e
