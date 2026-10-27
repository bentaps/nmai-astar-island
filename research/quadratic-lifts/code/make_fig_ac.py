import numpy as np
import allen_cahn as ac   # runs its own table first (cheap); reuse functions
from vizstyle import *
k = 0.05; T = 8.0; n = int(round(T/k)); t = k*np.arange(1, n+1)
series = {}
for name, method, gamma in [("Kahan-lift, unstabilised", 'kahan', 0.0), ("Kahan-lift, stabilised", 'kahan', 4.0), ("IEQ/SAV-type", 'ieq', None)]:
    u, q = ac.u0.copy(), ac.u0**2; uold = ac.u0.copy(); cs = []
    if method == 'ieq': q = ac.u0**2 - 1
    for _ in range(n):
        if method == 'kahan': u, q = ac.kahan_ac(u, q, k, gamma)
        else: un, q = ac.ieq(u, q, uold, k); uold = u; u = un
        c = (q - u**2) if method == 'kahan' else (q - (u**2 - 1))
        cs.append(np.max(abs(c)))
    series[name] = np.maximum(np.array(cs), 1e-16)
f, ax = fig()
for name, cs in series.items():
    ax.plot(t, cs, color=COL[name], lw=1.6, label=name)
setup(ax, "time t", "max constraint error |q − u²|", "Allen–Cahn, pointwise lift q = u², step k = 0.05")
ax.set_ylim(1e-12, 1e1); legend(ax, loc="center right")
ax.text(0.3, 2.0, "γ = 0: constraint manifold repels (c′ = 2(1−q)c)", color=INK2, fontsize=8.5)
ax.text(2.2, 1e-9, "γ = 4: c′ = (2(1−q)−γ)c, manifold attracts", color=INK2, fontsize=8.5)
f.tight_layout(); f.savefig("../figures/allen_cahn_constraint.png", dpi=160, facecolor=SURFACE)
print("ok")
