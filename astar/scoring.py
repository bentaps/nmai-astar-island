"""
Competition scoring: entropy-weighted KL divergence.

The Astar Island competition scores predictions by:

    weighted_kl = Σ_u H(p_u) · KL(p_u ‖ q_u) / Σ_u H(p_u)
    score = max(0, min(100, 100 · exp(−3 · weighted_kl)))

where p_u is the organizer's ground truth distribution at cell u,
q_u is our submitted distribution, H is Shannon entropy, and KL is
Kullback-Leibler divergence. Higher is better. 100 = perfect.

Static cells (mountains, ocean) have H(p) ≈ 0 and contribute nothing.
The score is dominated by high-entropy cells near settlements.
"""

import numpy as np


def entropy(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Shannon entropy H(p) = -Σ_c p_c ln(p_c).

    Args:
        p: (..., C) probability distributions (last axis = classes).
        eps: small constant to avoid log(0).

    Returns:
        (...) array of entropy values.
    """
    p_safe = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p_safe), axis=-1)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """KL divergence KL(p ‖ q) = Σ_c p_c ln(p_c / q_c).

    Args:
        p: (..., C) true distributions.
        q: (..., C) predicted distributions.
        eps: small constant to avoid log(0) and division by zero.

    Returns:
        (...) array of KL divergence values.
    """
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return np.sum(p * np.log(p_safe / q_safe), axis=-1)


def competition_score(p_gt: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Competition score: 100 · exp(−3 · Σ H(p_u)·KL(p_u‖q_u) / Σ H(p_u)).

    This is the canonical scorer. All evaluation must use this function.

    Higher is better. 100 = perfect prediction. 0 = terrible.
    Static cells (H ≈ 0) contribute nothing to the score.

    Args:
        p_gt: (H, W, C) ground truth probability tensor.
        q:    (H, W, C) predicted probability tensor.
        eps:  numerical stability constant.

    Returns:
        Scalar score in [0, 100].
    """
    h     = entropy(p_gt, eps=eps)           # (H, W)
    denom = float(h.sum())
    if denom < eps:
        return 100.0
    kl  = kl_divergence(p_gt, q, eps=eps)   # (H, W)
    wkl = float(np.sum(h * kl)) / denom
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * wkl))))


def entropy_weighted_kl(p_gt: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Alias for competition_score(). Higher is better, range [0, 100]."""
    return competition_score(p_gt, q, eps=eps)


def apply_floor_and_normalize(q: np.ndarray, floor: float = 0.01) -> np.ndarray:
    """Apply minimum probability floor and renormalize.

    Prevents zero probabilities which cause infinite KL divergence.

    Args:
        q: (..., C) probability distributions.
        floor: minimum probability per class.

    Returns:
        (..., C) floored and renormalized distributions.
    """
    q_floored = np.maximum(q, floor)
    q_floored = q_floored / q_floored.sum(axis=-1, keepdims=True)
    return q_floored
