import numpy as np
from vizstyle import *
from quartic_compare import KL, MQAV, SV, IEQ, STAG
H = lambda q, p: 0.5*p**2 + 0.25*q**4
q0, p0 = 1.0, 0.5; H0 = H(q0, p0); h = 0.1; T = 1e4; n = int(T/h); t = h*np.arange(n+1)
runs = {}
Q, P, Y = KL(q0, p0, h, n); runs["Kahan-lift"] = (Q, P, Y)
Q, P, _ = MQAV(q0, p0, h, n); runs["MQAV midpoint"] = (Q, P, None)
Q, P = SV(q0, p0, h, n); runs["Störmer–Verlet"] = (Q, P, None)
Q, P, Y = STAG(q0, p0, h, n); runs["Staggered"] = (Q, P, None)
Q, P, Y = IEQ(q0, p0, h, n); runs["IEQ/SAV-type"] = (Q, P, Y)
# --- Figure 1: relative energy error envelopes
f, ax = fig()
order = ["MQAV midpoint", "Kahan-lift", "Staggered", "Störmer–Verlet", "IEQ/SAV-type"]
for name in order:
    Q, P, _ = runs[name]; e = np.maximum(np.abs(H(Q, P) - H0)/H0, 1e-16)
    tt, ee = envelope(t, e)
    ax.plot(tt, ee, color=COL[name], lw=1.6, solid_capstyle="round", label=name)
setup(ax, "time t", "relative energy error |H − H₀|/H₀", "Quartic oscillator, h = 0.1, 10⁵ steps")
ax.set_ylim(1e-16, 1e2); legend(ax, loc="center right")
f.tight_layout(); f.savefig("../figures/quartic_energy.png", dpi=160, facecolor=SURFACE); 
# --- Figure 2: constraint error y - q^2 for the two LINEARLY IMPLICIT lifted methods that carry y
f, ax = fig()
for name in ["Kahan-lift", "IEQ/SAV-type"]:
    Q, P, Y = runs[name]; e = np.maximum(np.abs(Y - Q**2), 1e-16)
    tt, ee = envelope(t, e); ax.plot(tt, ee, color=COL[name], lw=1.6, label=name)
setup(ax, "time t", "constraint error |y − q²|", "Auxiliary-variable drift, h = 0.1")
ax.annotate("bounded: y − q² + h²p²/4 is conserved exactly", xy=(6000, 1.3e-3), xytext=(2500, 2e-5),
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("drifts: lifted energy conserved, constraint not", xy=(6000, 2.0), xytext=(1500, 60),
            color=INK2, fontsize=9, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_ylim(1e-6, 1e3); legend(ax, loc="lower right")
f.tight_layout(); f.savefig("../figures/quartic_constraint.png", dpi=160, facecolor=SURFACE)
print("ok")
