"""
Regime inference from settlement payloads.

Estimates round-level regime features (expansion, collapse, trade intensity)
from settlement metadata returned by the /simulate endpoint. These features
characterize the hidden simulation dynamics and can be used to adjust
calibration parameters per round.
"""

from __future__ import annotations

import numpy as np


def _initial_in_viewport(
    initial_settlements: list[dict],
    vp: dict,
) -> list[dict]:
    """Return initial settlements whose (x, y) falls within the viewport."""
    vx, vy = vp["x"], vp["y"]
    vw, vh = vp["w"], vp["h"]
    return [
        s for s in initial_settlements
        if vx <= s.get("x", -1) < vx + vw and vy <= s.get("y", -1) < vy + vh
    ]


def estimate_regime(
    settlement_log: list[dict],
    initial_settlements: list[list[dict]],
) -> dict:
    """Estimate round-level regime from settlement payload deltas.

    Uses viewport-local comparisons: for each query, compares the observed
    settlements against the initial settlements within the same viewport.
    This avoids the bias of comparing viewport-local counts to whole-seed totals.

    Args:
        settlement_log: enriched query log from _process_query_result().
            Each entry is a dict with keys: seed_idx, viewport, query_index,
            settlements, raw_grid.
        initial_settlements: list of initial settlement lists per seed
            (from round_data initial_states).

    Returns:
        dict with regime features:
            n_queries_with_data: how many queries had both observed and initial data
            initial_count:       mean initial settlement count across seeds
            observed_count:      mean observed settlement count per viewport
            expansion_rate:      viewport-local (observed - initial) / initial
            port_fraction:       fraction of observed settlements with ports
            mean_population:     mean population across all observed settlements
            mean_defense:        mean defense level
            owner_entropy:       Shannon entropy of owner_id distribution
            alive_fraction:      fraction of observed settlements that are alive
    """
    initial_counts = [len(s) for s in initial_settlements]
    mean_initial = float(np.mean(initial_counts)) if initial_counts else 0.0

    # Collect all observed settlements and compute viewport-local expansion
    all_settlements: list[dict] = []
    expansion_deltas: list[float] = []
    n_with_data = 0

    for entry in settlement_log:
        if isinstance(entry, dict):
            observed = entry.get("settlements", [])
            seed_idx = entry.get("seed_idx")
            vp = entry.get("viewport")
        else:
            # Legacy format: bare list of settlements (no viewport info)
            observed = entry
            seed_idx = None
            vp = None

        if not observed:
            continue

        n_with_data += 1
        all_settlements.extend(observed)

        # Viewport-local expansion rate (only when we have enriched metadata)
        if seed_idx is not None and vp is not None and seed_idx < len(initial_settlements):
            initial_in_vp = _initial_in_viewport(initial_settlements[seed_idx], vp)
            if initial_in_vp:
                delta = (len(observed) - len(initial_in_vp)) / len(initial_in_vp)
                expansion_deltas.append(delta)

    if n_with_data == 0:
        return {
            "n_queries_with_data": 0,
            "initial_count": mean_initial,
            "observed_count": 0.0,
            "expansion_rate": 0.0,
            "port_fraction": 0.0,
            "mean_population": 0.0,
            "mean_defense": 1.0,
            "owner_entropy": 0.0,
            "alive_fraction": 1.0,
        }

    # Expansion rate from viewport-local deltas
    if expansion_deltas:
        expansion_rate = float(np.mean(expansion_deltas))
    else:
        expansion_rate = 0.0

    # Port fraction
    n_ports = sum(1 for s in all_settlements if s.get("has_port", False))
    port_fraction = n_ports / len(all_settlements) if all_settlements else 0.0

    # Population stats
    pops = [s.get("population", 0.0) for s in all_settlements]
    mean_pop = float(np.mean(pops)) if pops else 0.0

    # Defense stats
    defs = [s.get("defense", 1.0) for s in all_settlements]
    mean_def = float(np.mean(defs)) if defs else 1.0

    # Alive fraction
    n_alive = sum(1 for s in all_settlements if s.get("alive", True))
    alive_frac = n_alive / len(all_settlements) if all_settlements else 1.0

    # Owner entropy (diversity of ownership)
    owner_ids = [s.get("owner_id", -1) for s in all_settlements]
    if owner_ids:
        _, counts = np.unique(owner_ids, return_counts=True)
        p = counts / counts.sum()
        owner_entropy = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        owner_entropy = 0.0

    mean_observed = len(all_settlements) / n_with_data

    return {
        "n_queries_with_data": n_with_data,
        "initial_count": mean_initial,
        "observed_count": mean_observed,
        "expansion_rate": float(np.clip(expansion_rate, -1.0, 5.0)),
        "port_fraction": port_fraction,
        "mean_population": mean_pop,
        "mean_defense": mean_def,
        "owner_entropy": owner_entropy,
        "alive_fraction": alive_frac,
    }


def regime_adjustments(regime: dict) -> dict:
    """Suggest calibration parameter adjustments based on inferred regime.

    Returns dict with keys: pool_weight_mult, floor_base_mult, temperature_adj.
    Currently returns identity — handcrafted thresholds were tested and found to
    hurt scores (round 6: -0.90, round 8: -0.04). Regime features are still
    computed and logged for future learned calibration.
    """
    # TODO: replace with learned adjustments once enough rounds with settlement
    # data are available for proper cross-validation
    return {"pool_weight_mult": 1.0, "floor_base_mult": 1.0, "temperature_adj": 0.0}


def regime_summary(regime: dict) -> str:
    """One-line human-readable regime summary."""
    if regime["n_queries_with_data"] == 0:
        return "No settlement data available"

    rate = regime["expansion_rate"]
    if rate > 0.5:
        phase = "heavy expansion"
    elif rate > 0.1:
        phase = "moderate expansion"
    elif rate > -0.1:
        phase = "stable"
    elif rate > -0.3:
        phase = "moderate decline"
    else:
        phase = "collapse"

    return (
        f"{phase} (exp={rate:+.0%}), "
        f"{regime['observed_count']:.0f} setts/query "
        f"(initial {regime['initial_count']:.0f}/seed), "
        f"pop={regime['mean_population']:.1f}, "
        f"ports={regime['port_fraction']:.0%}, "
        f"owners H={regime['owner_entropy']:.2f}"
    )
