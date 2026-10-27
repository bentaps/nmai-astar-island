"""Shared chart style (reference palette, light mode): fixed categorical order, method -> colour is fixed."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# colour follows the METHOD (entity) in every figure
COL = {"MQAV midpoint": "#2a78d6",          # slot 1 blue
       "Kahan-lift": "#eb6834",             # slot 2 orange
       "Staggered": "#1baf7a",              # slot 3 aqua
       "Störmer–Verlet": "#eda100",         # slot 4 yellow
       "IEQ/SAV-type": "#e87ba4",           # slot 5 magenta
       "Kahan-lift, stabilised": "#008300", # slot 6 green
       "Kahan-lift, unstabilised": "#6250d6"}  # slot 7 violet
def setup(ax, xlabel, ylabel, title=None, logy=True):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(AXIS); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-"); ax.set_axisbelow(True)
    if logy: ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=INK2, fontsize=10); ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    if title: ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
def fig(w=7.2, h=3.6, ncols=1):
    f, axs = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE)
    return f, axs
def envelope(t, e, nbins=300):
    """max |e| per time bin -> clean envelope for oscillatory error signals"""
    import numpy as np
    e = np.abs(e); idx = np.linspace(0, len(t), nbins+1).astype(int)
    tt = [t[idx[i+1]-1] for i in range(nbins) if idx[i+1] > idx[i]]
    ee = [e[idx[i]:idx[i+1]].max() for i in range(nbins) if idx[i+1] > idx[i]]
    return np.array(tt), np.array(ee)
def legend(ax, loc="best"):
    lg = ax.legend(frameon=False, fontsize=9, loc=loc, labelcolor=INK2, handlelength=1.6)
    return lg
