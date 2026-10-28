"""Figure 3: maximal relative energy error on [0, 200] against h*omega for the stiff isotropic oscillator in R^3
(H = |p|^2/2 + om^2 |q|^2/2 + |q|^4/4, om = 10, q0 = (0.8, 0.1, -0.3), p0 = (0, 3, 1)), for
  lifted Kahan (closed form, O(n) per step), Strang splitting with the exact harmonic flow in the order
  flow(h/2)-kick(h)-flow(h/2) (plotted; the kick-flow-kick "impulse" ordering is reported in the numbers only),
  the implicit midpoint rule (Newton) and Stormer-Verlet.
Also marks h*omega where D(z0) = 0 (Proposition 6.3). Writes ../fig_stepsize.pdf and ../fig_stepsize_numbers.txt."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import brentq
n, om = 3, 10.0; u2, u1 = 0.5, om**2/2; Up = lambda y: u2*y + u1
H = lambda q, p: p@p/2 + om**2*(q@q)/2 + (q@q)**2/4
q0 = np.array([0.8, 0.1, -0.3]); p0 = np.array([0.0, 3.0, 1.0]); E0 = H(q0, p0); T = 200.0; CAP = 1e2
def kahan_run(h):
    q, p = q0.copy(), p0.copy(); y = q@q - h*h/4*p@p; kap = h*h*u2; err = 0
    for k in range(int(round(T/h))):
        al = h*Up(y); Y = 1 + h*al/2; eta = q + h/2*p
        yn = (y*Y + h*(2*p - (al + h*u1)*q)@eta)/(Y + kap*(q@eta)); aln = h*Up(yn)
        q, p, y = ((1 - h*aln/2)*q + h*p)/Y, ((1 - h*al/2)*p - (al + aln)*q)/Y, yn
        e = abs(H(q, p) - E0)/E0
        if not np.isfinite(e) or e > CAP: return np.inf
        err = max(err, e)
    return err
def impulse_run(h):
    c, s = np.cos(om*h/2), np.sin(om*h/2); q, p = q0.copy(), p0.copy(); err = 0
    for k in range(int(round(T/h))):
        q, p = c*q + s/om*p, -om*s*q + c*p
        p = p - h*(q@q)*q
        q, p = c*q + s/om*p, -om*s*q + c*p
        e = abs(H(q, p) - E0)/E0
        if not np.isfinite(e) or e > CAP: return np.inf
        err = max(err, e)
    return err
def kfk_run(h):
    "impulse method in the kick-flow-kick ordering (half kicks), for comparison; not plotted"
    c, s = np.cos(om*h), np.sin(om*h); q, p = q0.copy(), p0.copy(); err = 0
    for k in range(int(round(T/h))):
        p = p - h/2*(q@q)*q
        q, p = c*q + s/om*p, -om*s*q + c*p
        p = p - h/2*(q@q)*q
        e = abs(H(q, p) - E0)/E0
        if not np.isfinite(e) or e > CAP: return np.inf
        err = max(err, e)
    return err
def verlet_run(h):
    q, p = q0.copy(), p0.copy(); err = 0
    for k in range(int(round(T/h))):
        p = p - h/2*(om**2 + q@q)*q; q = q + h*p; p = p - h/2*(om**2 + q@q)*q
        e = abs(H(q, p) - E0)/E0
        if not np.isfinite(e) or e > CAP: return np.inf
        err = max(err, e)
    return err
def midpoint_run(h):
    q, p = q0.copy(), p0.copy(); err = 0
    for k in range(int(round(T/h))):
        Q, P = q.copy(), p.copy()
        for it in range(50):
            qm, pm = (q + Q)/2, (p + P)/2; r2 = qm@qm
            G = np.concatenate([Q - q - h*pm, P - p + h*(om**2 + r2)*qm])
            Jg = np.block([[np.eye(n), -h/2*np.eye(n)], [h/2*((om**2 + r2)*np.eye(n) + 2*np.outer(qm, qm)), np.eye(n)]])
            d = np.linalg.solve(Jg, -G); Q += d[:n]; P += d[n:]
            if np.abs(d).max() < 1e-13: break
        q, p = Q, P; err = max(err, abs(H(q, p) - E0)/E0)
    return err
def D0(h):
    y = q0@q0 - h*h/4*p0@p0; K = p0@p0/2 + u2/2*(y + u1/u2)**2; return 1 + h*h*u2*(q0@q0) - h**4*u2/2*K
fine = np.linspace(0.2, 10.0, 1961); coarse = np.linspace(0.2, 10.0, 99)
K = np.array([kahan_run(x/om) for x in fine]); I = np.array([impulse_run(x/om) for x in fine])
Mp = np.array([midpoint_run(x/om) for x in coarse]); S = np.array([verlet_run(x/om) for x in fine if x < 2.1])
KFK = np.array([kfk_run(x/om) for x in fine])
hc = brentq(D0, 0.1, 0.3)*om
def bands(mask):
    "Group the grid points where mask holds into intervals [first, last] of consecutive points."
    idx, out = np.flatnonzero(mask), []
    for i in idx:
        if out and i == out[-1][1] + 1: out[-1][1] = i
        else: out.append([i, i])
    return ", ".join(f"[{fine[a]:.3f}, {fine[b]:.3f}]" for a, b in out)
lines = [f"D(z0) = 0 at h*om = {hc:.5f}",
         f"lifted Kahan: {int(np.sum(~np.isfinite(K)))} escapes of {len(fine)}; error > 0.3 or escape for h*om in "
         + bands((~np.isfinite(K)) | (K > 0.3)),
         f"impulse: {int(np.sum(~np.isfinite(I)))} escapes of {len(fine)}; error > 0.3 or escape for h*om in "
         + bands((~np.isfinite(I)) | (I > 0.3)),
         f"impulse, kick-flow-kick ordering (not plotted): {int(np.sum(~np.isfinite(KFK)))} escapes; error > 0.3 or "
         f"escape for {int(np.sum((~np.isfinite(KFK)) | (KFK > 0.3)))} step sizes, in " + bands((~np.isfinite(KFK)) | (KFK > 0.3)),
         f"median error over non-escaping runs, h*om in [2.2, 10]: Kahan {np.median(K[(fine > 2.2) & np.isfinite(K)]):.3f}, "
         f"impulse {np.median(I[(fine > 2.2) & np.isfinite(I)]):.3f}, kick-flow-kick {np.median(KFK[(fine > 2.2) & np.isfinite(KFK)]):.3f}, "
         f"midpoint {np.median(Mp[coarse > 2.2]):.4f}",
         f"Verlet: first escape at h*om = {fine[fine < 2.1][~np.isfinite(S)][0]:.3f}",
         "lifted Kahan in its band: " + ", ".join(f"h*om = {x:.3f}: {'escape' if not np.isfinite(K[np.argmin(abs(fine - x))]) else f'{K[np.argmin(abs(fine - x))]:.2f}'}"
                                                   for x in (1.995, 2.0, 2.005)),
         f"h*om <= 1: Kahan/Verlet error ratio {np.median(K[fine <= 1]/np.array([verlet_run(x/om) for x in fine[fine <= 1]])):.3f}"]
open("../fig_stepsize_numbers.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"
fig, ax = plt.subplots(figsize=(6.6, 2.6), constrained_layout=True)
top = 3.0
def show(x, v): return np.where(np.isfinite(v), v, top)
ax.semilogy(fine, show(fine, I), color=C2, lw=0.7, label="Strang splitting with exact harmonic flow")
ax.semilogy(fine, show(fine, K), color=C1, lw=0.9, label="lifted Kahan")
ax.semilogy(coarse, Mp, color=C3, lw=1.0, ls="--", label="implicit midpoint (Newton)")
ax.semilogy(fine[fine < 2.1], show(None, S), color=C4, lw=1.0, ls=":", label="Störmer–Verlet")
ax.axvline(hc, color="#8a8984", lw=0.6); ax.text(hc - 0.08, 3e-6, r"$D(z_0)=0$", fontsize=7, color=C4, ha="right")
for kk in range(1, 7): ax.axvline(kk*np.pi/2, color=C2, lw=0.3, ls=(0, (1, 3)))
ax.set_ylim(1e-6, top*1.05); ax.set_xlim(0, 10)
ax.set_xlabel(r"$h\omega$"); ax.set_ylabel(r"$\max_{t\leq 200}|H-H_0|/H_0$ (capped)")
ax.legend(frameon=False, loc="lower right", fontsize=7, ncol=2)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
fig.savefig("../fig_stepsize.pdf"); fig.savefig("../fig_stepsize.png", dpi=200); print("wrote ../fig_stepsize.pdf")
