"""Figure: the lifted Kahan map for the stiff isotropic anharmonic oscillator in R^3,
H = |p|^2/2 + om^2 |q|^2/2 + |q|^4/4  (U(y) = y^2/4 + om^2 y/2, lift y = |q|^2), om = 10.
(a) the modified invariants E_h, M_h, c_h are conserved to round-off over 10^4 steps at h*om = 5;
(b) relative energy error in time: lifted Kahan and implicit midpoint at h*om = 5, Stormer-Verlet at h*om = 1.5;
(c) maximal relative energy error on [0, 200] against h*om (Stormer-Verlet is unstable for h*om > 2).
Writes ../fig_isotropic.pdf and prints the numbers quoted in the paper."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
n, om = 3, 10.0; u2, u1 = 0.5, om**2/2; Up = lambda y: u2*y + u1
H = lambda q, p: p@p/2 + om**2*(q@q)/2 + (q@q)**2/4
q0 = np.array([0.8, 0.1, -0.3]); p0 = np.array([0.0, 3.0, 1.0]); E0 = H(q0, p0)
def kahan(q, p, y, h):                       # closed-form lifted Kahan step, O(n) work
    al = h*Up(y); D = 1 + h*al/2; a, b, c = q@q, q@p, p@p
    yn = (y*D - h*(al + h*u1)*(a + h*b/2) + 2*h*b + h**2*c)/(D + h**2*u2*(a + h*b/2))
    aln = h*Up(yn)
    return ((1 - h*aln/2)*q + h*p)/D, (-(al + aln)*q + (1 - h*al/2)*p)/D, yn
def midpoint(q, p, h):                        # implicit midpoint rule, Newton to 1e-14
    Q, P = q.copy(), p.copy(); its = 0
    for _ in range(50):
        qm, pm = (q + Q)/2, (p + P)/2; r2 = qm@qm
        G = np.concatenate([Q - q - h*pm, P - p + h*(om**2 + r2)*qm])
        Jg = np.block([[np.eye(n), -h/2*np.eye(n)], [h/2*((om**2 + r2)*np.eye(n) + 2*np.outer(qm, qm)), np.eye(n)]])
        d = np.linalg.solve(Jg, -G); Q += d[:n]; P += d[n:]; its += 1
        if np.abs(d).max() < 1e-14: break
    return Q, P, its
def verlet(q, p, h):
    p = p - h/2*(om**2 + q@q)*q; q = q + h*p; return q, p - h/2*(om**2 + q@q)*q
def invariants(q, p, y, h):
    M = np.cross(q, p); Mh = M/(1 + h**2/2*Up(y)); ch = y - q@q + h**2/4*p@p
    K = p@p/2 + u2*y**2/2 + u1*y + u1**2/(2*u2)
    Eh = (K + h**2*u2/2*Mh@Mh)/(1 + h**2*u2*(q@q) - h**4*u2/2*K)      # Theorem 6.1(b)
    return Eh, Mh, ch
def run_kahan(h, T):
    q, p = q0.copy(), p0.copy(); y = q@q - h**2/4*p@p   # start on the shifted sheet c_h = 0
    Eh0, Mh0, ch0 = invariants(q, p, y, h); out = []
    for k in range(int(round(T/h))):
        q, p, y = kahan(q, p, y, h); Eh, Mh, ch = invariants(q, p, y, h)
        out.append(((k + 1)*h, abs(Eh - Eh0)/abs(Eh0), np.abs(Mh - Mh0).max()/np.abs(Mh0).max(), abs(ch - ch0),
                    abs(H(q, p) - E0)/E0))
    return np.array(out)
def run_mid(h, T):
    q, p = q0.copy(), p0.copy(); out = []; its = 0
    for k in range(int(round(T/h))):
        q, p, i = midpoint(q, p, h); its += i; out.append(((k + 1)*h, abs(H(q, p) - E0)/E0))
    return np.array(out), its
def run_verlet(h, T):
    q, p = q0.copy(), p0.copy(); out = []
    for k in range(int(round(T/h))):
        q, p = verlet(q, p, h); e = abs(H(q, p) - E0)/E0
        if not np.isfinite(e) or e > 1e3: return np.array(out), True
        out.append(((k + 1)*h, e))
    return np.array(out), False
# (a) long-time conservation at h*om = 5
hA = 0.5; A = run_kahan(hA, 5000.0)
print(f"(a) h*om = 5, {len(A)} steps: max drift E_h {A[:,1].max():.1e}, M_h {A[:,2].max():.1e}, c_h {A[:,3].max():.1e};"
      f" max energy error {A[:,4].max():.2e}")
# (b) energy error in time
Kb = run_kahan(0.5, 200.0); Mb, itsb = run_mid(0.5, 200.0); Vb, blow = run_verlet(0.15, 200.0)
print(f"(b) Kahan h*om=5: max energy err {Kb[:,4].max():.2e}; midpoint h*om=5: {Mb[:,1].max():.2e}"
      f" ({itsb/len(Mb):.1f} Newton iterations per step); Verlet h*om=1.5: {Vb[:,1].max():.2e} (blow-up: {blow})")
# (c) max energy error against h*om
hs = om**-1*np.array([0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 1.9, 2.5, 3.5, 5.0, 7.0, 10.0])
rk, rm, rv = [], [], []
for h in hs:
    rk.append(run_kahan(h, 200.0)[:, 4].max()); M_, it_ = run_mid(h, 200.0); rm.append(M_[:, 1].max())
    V_, bl = run_verlet(h, 200.0); rv.append(np.nan if bl else V_[:, 1].max())
    print(f"(c) h*om={h*om:5.2f}: Kahan {rk[-1]:.2e}  midpoint {rm[-1]:.2e} ({it_/len(M_):.1f} its/step)  Verlet "
          + ("unstable" if bl else f"{rv[-1]:.2e}"))
# ---------------- figure ----------------
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e"})
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.15), constrained_layout=True)
eps = 1e-17
sub = slice(None, None, 10)
ax[0].semilogy(A[sub, 0], A[sub, 1] + eps, color=C1, lw=0.8, label=r"$E_h$")
ax[0].semilogy(A[sub, 0], A[sub, 2] + eps, color=C2, lw=0.8, ls="--", label=r"$M_h$")
ax[0].semilogy(A[sub, 0], A[sub, 3] + eps, color=C3, lw=0.8, ls=":", label=r"$c_h$")
ax[0].set_ylim(1e-17, 1e-11); ax[0].set_xlabel(r"$t$"); ax[0].set_ylabel("drift of modified invariant")
ax[0].legend(frameon=False, loc="upper left", ncol=3, handlelength=1.6, columnspacing=0.8)
ax[0].set_title(r"(a) lifted Kahan, $h\omega=5$", fontsize=8)
cm = np.maximum.accumulate
ax[1].semilogy(Kb[:, 0], cm(Kb[:, 4]), color=C1, lw=1.0, label=r"lifted Kahan, $h\omega=5$")
ax[1].semilogy(Mb[:, 0], cm(Mb[:, 1]), color=C2, lw=1.0, ls="--", label=r"midpoint, $h\omega=5$")
ax[1].semilogy(Vb[:, 0], cm(Vb[:, 1]), color=C3, lw=1.0, ls=":", label=r"Störmer–Verlet, $h\omega=1.5$")
ax[1].set_ylim(1e-4, 3); ax[1].set_xlabel(r"$t$"); ax[1].set_ylabel(r"$\max_{s\leq t}|H-H_0|/H_0$")
ax[1].legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=7)
ax[1].set_title("(b) energy error, running maximum", fontsize=8)
ax[2].loglog(hs*om, rk, color=C1, lw=1.0, marker="o", ms=3.5, label="lifted Kahan")
ax[2].loglog(hs*om, rm, color=C2, lw=1.0, ls="--", marker="s", ms=3.5, label="midpoint")
ax[2].loglog(hs*om, rv, color=C3, lw=1.0, ls=":", marker="^", ms=3.5, label=r"Störmer–Verlet")
ax[2].axvline(2.0, color="#8a8984", lw=0.6, ls="-"); ax[2].text(2.2, 1.2e-1, "Verlet\nunstable", fontsize=6.5, color="#52514e")
ax[2].set_xlabel(r"$h\omega$"); ax[2].set_ylabel(r"$\max_{t\leq 200}|H-H_0|/H_0$")
ax[2].legend(frameon=False, loc="lower right", handlelength=1.6, fontsize=7)
ax[2].set_title("(c) error against step size", fontsize=8)
for a in ax: a.tick_params(labelsize=7); a.spines[["top", "right"]].set_visible(False)
fig.savefig("../fig_isotropic.pdf"); fig.savefig("../fig_isotropic.png", dpi=200)
print("wrote ../fig_isotropic.pdf")
