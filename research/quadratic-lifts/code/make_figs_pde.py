import numpy as np
from vizstyle import *
d = np.load("kg_series.npz"); E0 = float(d["E0"])
f, axs = fig(w=7.6, h=3.4, ncols=2)
for ax, ratio, title in [(axs[0], "0.5", "h = dx/2 (all methods stable)"), (axs[1], "4.0", "h = 4·dx (beyond explicit CFL)")]:
    for name, key in [("MQAV midpoint", "MQAV"), ("Kahan-lift", "KL"), ("Staggered", "STAG"), ("Störmer–Verlet", "SV")]:
        k = f"{key}_{ratio}"
        if k + "_t" not in d: continue
        t, e = d[k + "_t"], np.maximum(np.abs(d[k + "_dE"])/E0, 1e-16)
        tt, ee = envelope(t[1:], e[1:], nbins=200)
        ax.plot(tt, ee, color=COL[name], lw=1.6, label=name, zorder={"Kahan-lift": 4, "MQAV midpoint": 3, "Staggered": 2, "Störmer–Verlet": 1}[name])
    setup(ax, "time t", "relative energy error" if ratio == "0.5" else "", title)
    ax.set_ylim(1e-16, 1e0)
axs[1].text(500, 1e-6, "Staggered and Störmer–Verlet\nblow up at this step size", color=INK2, fontsize=8.5, ha="center")
legend(axs[0], loc="center right")
axs[0].text(30, 6e-3, "Kahan-lift ≈ Störmer–Verlet here", color=INK2, fontsize=8.5)
f.suptitle("φ⁴ Klein–Gordon, N = 512, T = 1000", color=INK, fontsize=11, x=0.02, ha="left")
f.tight_layout(); f.savefig("../figures/kg_energy.png", dpi=160, facecolor=SURFACE)
print("ok")
