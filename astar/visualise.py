"""
Visualisation utilities for Astar Island round data.

Provides colour maps, terrain/prediction renderers, entropy and coverage maps,
and matplotlib legend helpers. Import these in notebooks and scripts.

    from astar.visualise import (
        CLASS_COLORS, CLASS_NAMES, terrain_to_rgb, argmax_to_rgb,
        entropy_map, observation_count_map, legend_patches,
    )
"""

from __future__ import annotations

import matplotlib.patches as mpatches
import numpy as np

# RGB colours for the 6 competition classes
CLASS_COLORS = np.array([
    [0.55, 0.75, 0.95],   # 0 Empty / Ocean / Plains — light blue
    [0.95, 0.75, 0.20],   # 1 Settlement — gold
    [0.15, 0.55, 0.90],   # 2 Port — deep blue
    [0.65, 0.35, 0.20],   # 3 Ruin — brown
    [0.20, 0.65, 0.25],   # 4 Forest — green
    [0.60, 0.60, 0.60],   # 5 Mountain — grey
])

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# 8-code terrain → 6-class mapping
TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}


def terrain_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Convert an 8-code terrain grid (H, W) to an RGB image (H, W, 3)."""
    H, W = grid.shape
    rgb = np.zeros((H, W, 3))
    for code, cls in TERRAIN_TO_CLASS.items():
        rgb[grid == code] = CLASS_COLORS[cls]
    return rgb


def argmax_to_rgb(pred: np.ndarray) -> np.ndarray:
    """Convert a (H, W, 6) probability tensor to an argmax RGB image (H, W, 3)."""
    return CLASS_COLORS[pred.argmax(axis=-1)]


def entropy_map(pred: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-cell Shannon entropy of a (H, W, 6) probability tensor → (H, W)."""
    p = np.clip(pred, eps, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def observation_count_map(obs: np.ndarray) -> np.ndarray:
    """Total observations per cell from a (H, W, 6) count array → (H, W)."""
    return obs.sum(axis=-1)


def legend_patches() -> list:
    """Return a list of matplotlib Patch objects for the 6 classes."""
    return [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(6)
    ]


def plot_class_maps(
    pred:       np.ndarray,
    gt:         np.ndarray,
    pred_label: str = "Prediction",
    title:      str = "",
    save_path=None,
    show:       bool = True,
) -> None:
    """Plot per-class probability heatmaps: pred (row 0) vs ground truth (row 1).

    Args:
        pred:       (H, W, 6) probability array to show in the top row.
        gt:         (H, W, 6) ground truth probability array.
        pred_label: row 0 label (e.g. "Prior", "Prediction").
        title:      figure suptitle — typically includes seed index + score.
        save_path:  if given, save figure to this path before showing.
        show:       if False, suppress plt.show() (caller manages display).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(22, 8))
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    for c in range(6):
        ax = axes[0, c]
        im = ax.imshow(pred[:, :, c], origin="upper", interpolation="nearest",
                       cmap="gnuplot", vmin=0, vmax=1)
        ax.set_title(f"{pred_label}  P({CLASS_NAMES[c]})", fontsize=10,
                     color=CLASS_COLORS[c] * 0.7)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7, format="%.2f")

        ax2 = axes[1, c]
        im2 = ax2.imshow(gt[:, :, c], origin="upper", interpolation="nearest",
                         cmap="gnuplot", vmin=0, vmax=1)
        ax2.set_title(f"GT  P({CLASS_NAMES[c]})", fontsize=10)
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, shrink=0.7, format="%.2f")

    axes[0, 0].text(-0.05, 0.5, pred_label, transform=axes[0, 0].transAxes,
                    fontsize=11, fontweight="bold", va="center", rotation=90)
    axes[1, 0].text(-0.05, 0.5, "Ground truth", transform=axes[1, 0].transAxes,
                    fontsize=11, fontweight="bold", va="center", rotation=90)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
