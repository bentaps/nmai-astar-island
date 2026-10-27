"""Figure: degree growth d_k of the k-th iterate of lifted Kahan maps (restricted to a random line, exact modulo a
prime). The sequences are the outputs of ../../code-applications/integrability_degrees.py,
../../code-applications/isotropic_variants_degrees.py and ../../code-theory/degree_cases.py (see the README there).
Writes ../fig_degrees.pdf."""
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
integrable = {
    r"quartic oscillator ($\mathbb{R}^3$)": [3, 9, 19, 33, 51, 73, 99],
    r"isotropic quartic, $n=2$ ($\mathbb{R}^5$)": [4, 12, 25, 44, 68, 97, 132],
    "isotropic, magnetic field": [4, 13, 28, 49, 76, 109],
    r"decoupled $q_1^4+q_2^4$": [5, 17, 37, 65, 101, 145],
}
chaotic = {
    r"anisotropic $(q_1^2+2q_2^2)^2$": [4, 16, 58, 208, 742],
    "Garnier": [4, 16, 58, 208],
    "open Toda, 3 particles": [3, 9, 23, 57, 139, 337, 815],
    "sextic isotropic": [5, 16, 43, 114, 305],
}
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.linewidth": 0.6,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
C1, C2 = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(4.6, 2.9), constrained_layout=True)
mk = ["o", "s", "D", "^"]
for (name, d), m in zip(integrable.items(), mk):
    ax.semilogy(range(1, len(d) + 1), d, color=C1, lw=1.0, marker=m, ms=3.5, label=name)
for (name, d), m in zip(chaotic.items(), mk):
    ax.semilogy(range(1, len(d) + 1), d, color=C2, lw=1.0, ls="--", marker=m, ms=3.5, mfc="white", label=name)
ax.set_xlabel(r"iterate $k$"); ax.set_ylabel(r"degree $d_k$"); ax.set_xlim(0.8, 7.3)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
h, l = ax.get_legend_handles_labels()
leg1 = ax.legend(h[:4], l[:4], title="integrable (quadratic growth)", loc="upper left", frameon=False, fontsize=6.5,
                 title_fontsize=7, handlelength=2.2)
leg1._legend_box.align = "left"; ax.add_artist(leg1)
leg2 = ax.legend(h[4:], l[4:], title="not integrable (exponential growth)", loc="lower right", frameon=False,
                 fontsize=6.5, title_fontsize=7, handlelength=2.2)
leg2._legend_box.align = "left"
fig.savefig("../fig_degrees.pdf"); fig.savefig("../fig_degrees.png", dpi=200); print("wrote ../fig_degrees.pdf")
