"""
Active inference engine: query planning, observation collection, prediction building.

Works with any session that implements .simulate(seed_idx, x, y, w, h) -> dict,
i.e. both LiveSession (astar.api) and SimulatedSession (astar.session).
"""

from __future__ import annotations

import numpy as np

from .features import (
    NUM_CLASSES,
    NUM_RAW_CODES,
    RAW_CODE_TO_IDX,
    TERRAIN_TO_CLASS,
    collapse_8state_to_6class,
    compute_cell_features,
    distance_to_nearest_settlement,
    DEFAULT_DIST_BINS,
    DEFAULT_DENSITY_BINS,
    DEFAULT_DENSITY_RADIUS,
)
from .scoring import apply_floor_and_normalize


# ---------------------------------------------------------------------------
# Query planning
# ---------------------------------------------------------------------------

def full_coverage_tiling(H: int = 40, W: int = 40, vp: int = 15) -> list[tuple]:
    """Generate viewport positions for full map coverage.

    Produces a 3×3 grid of 15×15 viewports (with edge overlap) that
    covers the entire map using exactly 9 queries per seed.

    Returns:
        List of (x, y, w, h) tuples.
    """
    def tile_starts(dim: int, vp_size: int) -> list[int]:
        starts = [0]
        pos = 0
        while pos + vp_size < dim:
            pos = min(pos + vp_size, dim - vp_size)
            if pos not in starts:
                starts.append(pos)
        return starts

    viewports = []
    for y in tile_starts(H, vp):
        for x in tile_starts(W, vp):
            viewports.append((x, y, min(vp, W - x), min(vp, H - y)))
    return viewports


def rank_windows_by_entropy(
    alpha_post: np.ndarray,
    vp: int = 15,
    stride: int = 5,
) -> list[tuple]:
    """Rank candidate windows by total posterior entropy (highest first).

    Targets resampling queries where reducing uncertainty matters most.

    Args:
        alpha_post: (H, W, 6) posterior Dirichlet pseudo-counts.
        vp:         viewport size.
        stride:     step size for candidate window positions.

    Returns:
        Sorted list of (entropy_score, x, y, w, h).
    """
    q = alpha_post / alpha_post.sum(axis=-1, keepdims=True)
    q_safe = np.clip(q, 1e-12, 1.0)
    cell_entropy = -np.sum(q * np.log(q_safe), axis=-1)   # (H, W)

    H, W = cell_entropy.shape
    scored = []
    for y in range(0, max(H - vp + 1, 1), stride):
        for x in range(0, max(W - vp + 1, 1), stride):
            w = min(vp, W - x)
            h = min(vp, H - y)
            scored.append((float(cell_entropy[y:y+h, x:x+w].sum()), x, y, w, h))

    scored.sort(reverse=True)
    return scored


def rank_windows_by_dynamism(
    terrain: np.ndarray,
    settlements: list[dict],
    vp: int = 15,
    stride: int = 5,
) -> list[tuple]:
    """Rank candidate windows by expected dynamism (heuristic fallback).

    Scores windows higher if they contain many settlements, coastal cells,
    or other dynamic terrain types.

    Args:
        terrain:     (H, W) terrain code grid.
        settlements: list of settlement dicts.
        vp:          viewport size.
        stride:      step size for candidate window positions.

    Returns:
        Sorted list of (score, x, y, w, h), highest first.
    """
    H, W = terrain.shape
    settlement_positions = {(s["x"], s["y"]) for s in settlements}
    coastal_positions: set[tuple] = set()

    for y in range(H):
        for x in range(W):
            if terrain[y, x] != 10:   # not ocean
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and terrain[ny, nx] == 10:
                        coastal_positions.add((x, y))
                        break

    scored = []
    for y in range(0, H - vp + 1, stride):
        for x in range(0, W - vp + 1, stride):
            w = min(vp, W - x)
            h = min(vp, H - y)
            score = 0.0
            for wy in range(y, y + h):
                for wx in range(x, x + w):
                    if (wx, wy) in settlement_positions:
                        score += 10.0
                    if (wx, wy) in coastal_positions:
                        score += 2.0
                    code = terrain[wy, wx]
                    if code in (1, 2, 3):
                        score += 5.0
                    elif code == 11:
                        score += 0.5
                    elif code == 4:
                        score += 0.3
            scored.append((score, x, y, w, h))

    scored.sort(reverse=True)
    return scored


def dynamic_coverage_viewports(
    terrain: np.ndarray,
    settlements: list[dict],
    vp: int = 15,
    stride: int = 5,
    max_settlement_dist: int = 12,
    max_viewports: int | None = None,
) -> list[tuple]:
    """Generate viewports that maximize coverage of dynamic cells.

    Uses greedy set-cover: repeatedly picks the viewport covering
    the most uncovered dynamic cells. Dynamic cells are non-static
    terrain within max_settlement_dist of a settlement.

    Args:
        terrain:             (H, W) terrain code grid.
        settlements:         list of settlement dicts with 'x', 'y'.
        vp:                  viewport size.
        stride:              step size for candidate positions.
        max_settlement_dist: cells farther than this are considered static.
        max_viewports:       maximum number of viewports to return.

    Returns:
        List of (x, y, w, h) tuples in selection order (best first).
    """
    H, W = terrain.shape
    dist = distance_to_nearest_settlement(terrain, settlements)
    static_codes = {5, 10}  # mountain, ocean
    dynamic = np.zeros((H, W), dtype=bool)
    for y in range(H):
        for x in range(W):
            if terrain[y, x] not in static_codes and dist[y, x] <= max_settlement_dist:
                dynamic[y, x] = True

    # Generate candidate viewports
    candidates = []
    for cy in range(0, max(H - vp + 1, 1), stride):
        for cx in range(0, max(W - vp + 1, 1), stride):
            w = min(vp, W - cx)
            h = min(vp, H - cy)
            count = int(dynamic[cy:cy+h, cx:cx+w].sum())
            candidates.append((count, cx, cy, w, h))

    # Greedy set-cover
    covered = np.zeros((H, W), dtype=bool)
    selected = []
    candidates.sort(reverse=True)

    limit = max_viewports if max_viewports is not None else len(candidates)
    for _ in range(limit):
        best_gain, best_vp = -1, None
        for count, cx, cy, w, h in candidates:
            uncovered_dynamic = dynamic[cy:cy+h, cx:cx+w] & ~covered[cy:cy+h, cx:cx+w]
            gain = int(uncovered_dynamic.sum())
            if gain > best_gain:
                best_gain = gain
                best_vp = (cx, cy, w, h)
        if best_gain <= 0 or best_vp is None:
            break
        cx, cy, w, h = best_vp
        covered[cy:cy+h, cx:cx+w] = True
        selected.append(best_vp)
        candidates = [(c, x, y, w2, h2) for c, x, y, w2, h2 in candidates
                       if (x, y, w2, h2) != best_vp]

    return selected


# ---------------------------------------------------------------------------
# Adaptive multi-pass query strategy
# ---------------------------------------------------------------------------

def run_adaptive_queries(
    session,
    num_seeds: int,
    terrains: list[np.ndarray],
    settlements_list: list[list[dict]],
    alphas: list[np.ndarray],
    budget: int = 50,
    max_settlement_dist: int = 12,
    pool_weight: float = 1.5,
    temperature: float = 1.0,
    floor_base: float = 0.002,
    floor_obs_scale: float = 0.02,
    map_size: tuple[int, int] = (40, 40),
    verbose: bool = True,
) -> tuple[dict, dict, list[dict]]:
    """Adaptive multi-pass query strategy: diagnose, spread, refine.

    Phase 1 (Diagnostic): Query 2 diagnostic seeds thoroughly (~40% budget).
        Cross-seed pooling propagates information to all seeds.
    Phase 2 (Spread):     Query remaining seeds with targeted viewports (~40% budget).
        Focus on cells where pooled predictions are most uncertain.
    Phase 3 (Refine):     Spend remaining budget on highest-entropy windows (~20% budget).

    Args:
        session:            LiveSession or SimulatedSession.
        num_seeds:          number of seeds.
        terrains:           list of (H, W) terrain grids per seed.
        settlements_list:   list of settlement lists per seed.
        alphas:             list of (H, W, 6) Dirichlet priors per seed.
        budget:             total query budget.
        max_settlement_dist: dynamic cell distance threshold.
        pool_weight:        cross-seed pooling strength.
        temperature:        posterior temperature.
        floor_base:         dynamic floor base.
        floor_obs_scale:    dynamic floor observation scale.
        map_size:           (H, W) of the full map.
        verbose:            print progress.

    Returns:
        observations:   dict[seed_idx] -> (H, W, NUM_CLASSES) total obs counts.
        extra_obs:      dict[seed_idx] -> (H, W, NUM_CLASSES) refinement counts.
        settlement_log: list of enriched query records from all queries.
    """
    H, W = map_size
    settlement_log: list[dict] = []

    # Budget allocation
    diag_budget   = max(2, int(budget * 0.40))
    spread_budget = max(2, int(budget * 0.40))
    refine_budget = budget - diag_budget - spread_budget

    # Pick 2 diagnostic seeds (most dynamic cells — likely most informative)
    dynamic_counts = []
    for s in range(num_seeds):
        vps = dynamic_coverage_viewports(
            terrains[s], settlements_list[s],
            max_settlement_dist=max_settlement_dist,
            max_viewports=100,
        )
        dynamic_counts.append((len(vps), s))
    dynamic_counts.sort(reverse=True)
    diag_seeds = [s for _, s in dynamic_counts[:2]]
    other_seeds = [s for s in range(num_seeds) if s not in diag_seeds]

    if verbose:
        print(f"\n  Adaptive strategy: diag={diag_budget}, "
              f"spread={spread_budget}, refine={refine_budget}")
        print(f"  Diagnostic seeds: {diag_seeds}  "
              f"(most dynamic cells)")

    # ── Phase 1: Diagnostic — thorough coverage of 2 seeds ──
    observations = {i: np.zeros((H, W, NUM_RAW_CODES), dtype=float)
                    for i in range(num_seeds)}
    queries_used = 0
    diag_per_seed = diag_budget // len(diag_seeds)

    if verbose:
        print(f"\n--- Adaptive Phase 1: Diagnostic "
              f"({diag_per_seed} vps × {len(diag_seeds)} seeds) ---")

    for s in diag_seeds:
        vps = dynamic_coverage_viewports(
            terrains[s], settlements_list[s],
            max_viewports=diag_per_seed,
            max_settlement_dist=max_settlement_dist,
        )
        for vp_idx, (vx, vy, vw, vh) in enumerate(vps):
            if verbose:
                print(f"  Seed {s}, vp {vp_idx+1}/{len(vps)}: "
                      f"({vx},{vy}) {vw}×{vh}", end="")
            result = session.simulate(s, vx, vy, vw, vh)
            _process_query_result(result, observations[s], settlement_log,
                                  seed_idx=s, query_index=queries_used)
            queries_used += 1
            if verbose:
                remaining = result.get("queries_max", budget) - result.get("queries_used", queries_used)
                print(f" — budget remaining: {remaining}")

    # ── Phase 2: Spread — targeted queries on remaining seeds ──
    # Build interim pooled predictions to identify where uncertainty is highest
    interim_preds = build_predictions_pooled(
        alphas, observations, terrains, settlements_list,
        pool_weight=pool_weight, temperature=temperature,
        floor_base=floor_base, floor_obs_scale=floor_obs_scale,
    )

    spread_per_seed = spread_budget // max(len(other_seeds), 1)
    if verbose:
        print(f"\n--- Adaptive Phase 2: Spread "
              f"({spread_per_seed} vps × {len(other_seeds)} seeds) ---")

    for s in other_seeds:
        # Use entropy-based targeting: rank windows by pooled posterior entropy
        alpha_post_s = alphas[s] + observations[s]
        q_interim = interim_preds[s]
        q_safe = np.clip(q_interim, 1e-12, 1.0)
        cell_entropy = -np.sum(q_interim * np.log(q_safe), axis=-1)

        # Rank candidate windows by total entropy (highest uncertainty first)
        ranked = rank_windows_by_entropy(alpha_post_s)
        vps = [(x, y, w, h) for _, x, y, w, h in ranked[:spread_per_seed]]
        for vp_idx, (vx, vy, vw, vh) in enumerate(vps):
            if verbose:
                ent = float(cell_entropy[vy:vy+vh, vx:vx+vw].sum())
                print(f"  Seed {s}, vp {vp_idx+1}/{len(vps)}: "
                      f"({vx},{vy}) {vw}×{vh} [ent={ent:.1f}]", end="")
            result = session.simulate(s, vx, vy, vw, vh)
            _process_query_result(result, observations[s], settlement_log,
                                  seed_idx=s, query_index=queries_used)
            queries_used += 1
            if verbose:
                remaining = result.get("queries_max", budget) - result.get("queries_used", queries_used)
                print(f" — budget remaining: {remaining}")

    # ── Phase 3: Refine — entropy-based resampling across all seeds ──
    # Use ALL remaining budget (not just pre-allocated refine_budget)
    actual_remaining = budget - queries_used
    refine_budget = actual_remaining

    if verbose:
        print(f"\n--- Adaptive Phase 3: Refine "
              f"({refine_budget} queries) ---")

    extra_obs = {i: np.zeros((H, W, NUM_RAW_CODES), dtype=float)
                 for i in range(num_seeds)}

    if refine_budget > 0:
        alphas_post = [alphas[s] + observations[s] for s in range(num_seeds)]
        extra_obs, _ = run_resampling_queries(
            session, num_seeds, terrains, settlements_list, refine_budget,
            alphas_post=alphas_post, map_size=map_size,
            verbose=verbose, settlement_log=settlement_log,
        )

    return observations, extra_obs, settlement_log


# ---------------------------------------------------------------------------
# Cross-seed feature pooling
# ---------------------------------------------------------------------------

def pool_observations_by_feature(
    observations: dict[int, np.ndarray],
    terrains: list[np.ndarray],
    settlements_list: list[list[dict]],
    dist_bins: np.ndarray = DEFAULT_DIST_BINS,
    density_bins: np.ndarray = DEFAULT_DENSITY_BINS,
    density_radius: int = DEFAULT_DENSITY_RADIUS,
) -> dict[tuple, np.ndarray]:
    """Pool observation counts by feature key across all seeds.

    Groups observed cells by their feature key (terrain_code, dist_bin,
    is_coastal, density_bin) and sums their observation counts. This gives
    ~20-100 effective observations per bin instead of 1 per cell.

    Args:
        observations:     dict[seed_idx] -> (H, W, NUM_CLASSES) count array.
        terrains:         list of (H, W) terrain grids per seed.
        settlements_list: list of settlement lists per seed.

    Returns:
        dict mapping feature_key tuple -> (NUM_RAW_CODES,) total counts (8-state).
    """
    pooled: dict[tuple, np.ndarray] = {}

    for seed_idx, obs in observations.items():
        terrain = terrains[seed_idx]
        setts = settlements_list[seed_idx]
        tc_map, db_map, _, co_map, den_map = compute_cell_features(
            terrain, setts, dist_bins, density_bins, density_radius,
        )
        H, W = terrain.shape
        obs_total = obs.sum(axis=-1)  # (H, W) — nonzero where observed

        for y in range(H):
            for x in range(W):
                if obs_total[y, x] < 0.5:
                    continue
                key = (int(tc_map[y, x]), int(db_map[y, x]),
                       int(co_map[y, x]), int(den_map[y, x]))
                if key not in pooled:
                    pooled[key] = np.zeros(NUM_RAW_CODES, dtype=float)
                pooled[key] += obs[y, x]

    return pooled


# ---------------------------------------------------------------------------
# Observation collection
# ---------------------------------------------------------------------------

def run_queries(
    session,
    seed_idx: int,
    viewports: list[tuple],
    map_size: tuple[int, int] = (40, 40),
) -> np.ndarray:
    """Run a list of viewports for one seed, accumulate observation counts.

    Works with any session that has .simulate(seed_idx, x, y, w, h) -> dict.

    Args:
        session:   LiveSession or SimulatedSession.
        seed_idx:  which seed to query.
        viewports: list of (x, y, w, h) tuples.
        map_size:  (H, W) of the full map.

    Returns:
        (H, W, NUM_RAW_CODES) observation count array (8-state).
    """
    H, W = map_size
    obs = np.zeros((H, W, NUM_RAW_CODES), dtype=float)
    for vx, vy, vw, vh in viewports:
        result = session.simulate(seed_idx, vx, vy, vw, vh)
        vp = result["viewport"]
        for ri, row in enumerate(result["grid"]):
            for ci, code in enumerate(row):
                gy, gx = vp["y"] + ri, vp["x"] + ci
                obs[gy, gx, RAW_CODE_TO_IDX.get(code, 0)] += 1.0
    return obs


def _process_query_result(
    result: dict,
    obs: np.ndarray,
    settlement_log: list[dict] | None = None,
    seed_idx: int | None = None,
    query_index: int | None = None,
) -> None:
    """Accumulate observation counts from a single query result.

    Modifies obs in-place (8-state). Optionally appends enriched query record
    containing seed_idx, viewport, query_index, settlements, and raw_grid.
    """
    vp = result["viewport"]
    for ri, row in enumerate(result["grid"]):
        for ci, code in enumerate(row):
            gy, gx = vp["y"] + ri, vp["x"] + ci
            obs[gy, gx, RAW_CODE_TO_IDX.get(code, 0)] += 1.0
    if settlement_log is not None:
        settlement_log.append({
            "seed_idx": seed_idx,
            "viewport": vp,
            "query_index": query_index,
            "settlements": result.get("settlements", []),
            "raw_grid": result["grid"],
        })


def run_coverage_queries(
    session,
    num_seeds: int,
    viewports: list[tuple] | list[list[tuple]],
    map_size: tuple[int, int] = (40, 40),
    verbose: bool = True,
) -> tuple[dict, int, list[dict]]:
    """Run coverage tiling for all seeds.

    Args:
        session:    LiveSession or SimulatedSession.
        num_seeds:  number of seeds to query.
        viewports:  either a shared list of (x, y, w, h) for all seeds,
                    or a list of per-seed viewport lists.
        map_size:   (H, W) of the full map.
        verbose:    print progress.

    Returns:
        observations:    dict[seed_idx] -> (H, W, NUM_RAW_CODES) count array (8-state).
        queries_used:    total queries consumed.
        settlement_log:  list of enriched query records from each query.
    """
    H, W = map_size
    observations: dict[int, np.ndarray] = {}
    queries_used = 0
    settlement_log: list[dict] = []

    # Determine if viewports is per-seed or shared
    per_seed = (isinstance(viewports, list) and len(viewports) > 0
                and isinstance(viewports[0], list))

    for seed_idx in range(num_seeds):
        seed_vps = viewports[seed_idx] if per_seed else viewports
        obs = np.zeros((H, W, NUM_RAW_CODES), dtype=float)
        for vp_idx, (vx, vy, vw, vh) in enumerate(seed_vps):
            if verbose:
                print(f"  Seed {seed_idx}, vp {vp_idx+1}/{len(seed_vps)}: "
                      f"({vx},{vy}) {vw}×{vh}", end="")
            result = session.simulate(seed_idx, vx, vy, vw, vh)
            _process_query_result(result, obs, settlement_log,
                                  seed_idx=seed_idx, query_index=queries_used)
            queries_used += 1
            if verbose:
                remaining = result.get("queries_max", 50) - result.get("queries_used", queries_used)
                print(f" — budget remaining: {remaining}")
        observations[seed_idx] = obs

    return observations, queries_used, settlement_log


def run_resampling_queries(
    session,
    num_seeds: int,
    terrains: list[np.ndarray],
    settlements_list: list[list[dict]],
    budget: int,
    alphas_post: list[np.ndarray] | None = None,
    map_size: tuple[int, int] = (40, 40),
    verbose: bool = True,
    settlement_log: list[dict] | None = None,
) -> tuple[dict, int]:
    """Spend remaining budget on highest-value resampling windows.

    Uses entropy-based ranking when posterior alphas are available,
    otherwise falls back to dynamism heuristics.

    Args:
        session:          LiveSession or SimulatedSession.
        num_seeds:        number of seeds.
        terrains:         list of (H, W) terrain grids per seed.
        settlements_list: list of settlement lists per seed.
        budget:           number of queries available.
        alphas_post:      optional list of (H, W, 6) posterior alphas per seed.
        map_size:         (H, W) of the full map.
        verbose:          print progress.
        settlement_log:   if provided, append settlement payloads from each query.

    Returns:
        extra_obs:    dict[seed_idx] -> (H, W, NUM_RAW_CODES) extra counts (8-state).
        queries_used: total queries consumed.
    """
    if budget <= 0:
        return {i: np.zeros((*map_size, NUM_RAW_CODES)) for i in range(num_seeds)}, 0

    H, W = map_size
    extra_obs = {i: np.zeros((H, W, NUM_RAW_CODES), dtype=float) for i in range(num_seeds)}
    queries_used = 0

    candidates = []
    for seed_idx in range(num_seeds):
        if alphas_post is not None:
            ranked = rank_windows_by_entropy(alphas_post[seed_idx])
            label = "entropy"
        else:
            ranked = rank_windows_by_dynamism(terrains[seed_idx], settlements_list[seed_idx])
            label = "dynamism"
        for score, x, y, w, h in ranked[:3]:
            candidates.append((score, seed_idx, x, y, w, h, label))

    candidates.sort(reverse=True)

    for score, seed_idx, vx, vy, vw, vh, label in candidates[:budget]:
        if verbose:
            print(f"  Resample: seed {seed_idx}, ({vx},{vy}) {vw}×{vh} "
                  f"[{label}={score:.1f}]", end="")
        result = session.simulate(seed_idx, vx, vy, vw, vh)
        _process_query_result(result, extra_obs[seed_idx], settlement_log,
                              seed_idx=seed_idx, query_index=queries_used)
        queries_used += 1
        if verbose:
            remaining = result.get("queries_max", 50) - result.get("queries_used", queries_used)
            print(f" — budget remaining: {remaining}")

    return extra_obs, queries_used


# ---------------------------------------------------------------------------
# Prediction building
# ---------------------------------------------------------------------------

def build_predictions(
    alphas: list[np.ndarray],
    observations: dict[int, np.ndarray],
    extra_obs: dict[int, np.ndarray] | None = None,
    floor: float = 0.01,
) -> list[np.ndarray]:
    """Fuse Dirichlet priors with empirical observation counts.

    posterior α = prior α + observation counts
    prediction   = posterior mean, floored and renormalized

    Args:
        alphas:       list of (H, W, 8) Dirichlet prior pseudo-counts per seed.
        observations: dict[seed_idx] -> (H, W, 8) observation counts.
        extra_obs:    optional dict[seed_idx] -> (H, W, 8) additional counts.
        floor:        minimum probability per class.

    Returns:
        list of (H, W, 6) normalized probability tensors per seed (collapsed).
    """
    predictions = []
    for seed_idx, alpha_prior in enumerate(alphas):
        alpha_post = alpha_prior.copy()
        if seed_idx in observations:
            alpha_post += observations[seed_idx]
        if extra_obs and seed_idx in extra_obs:
            alpha_post += extra_obs[seed_idx]
        q8 = alpha_post / alpha_post.sum(axis=-1, keepdims=True)
        q = collapse_8state_to_6class(q8)
        predictions.append(apply_floor_and_normalize(q, floor=floor))
    return predictions


def dynamic_floor(
    alpha_post: np.ndarray,
    alpha_prior: np.ndarray,
    floor_base: float = 0.002,
    floor_obs_scale: float = 0.02,
) -> np.ndarray:
    """Per-cell probability floor based on posterior evidence.

    floor(u) = floor_base + floor_obs_scale / (1 + n_obs(u))

    Cells with more evidence (higher α_post − α_prior) get a tighter floor,
    allowing sharper, more trusted predictions. Unobserved cells get a wider
    floor for safer smoothing.

    Args:
        alpha_post:      (H, W, C) posterior pseudo-counts.
        alpha_prior:     (H, W, C) prior pseudo-counts.
        floor_base:      minimum floor even for well-observed cells.
        floor_obs_scale: additional floor for unobserved cells.

    Returns:
        (H, W, 1) per-cell floor values for broadcasting against (H, W, C).
    """
    n_obs = (alpha_post - alpha_prior).clip(min=0).sum(axis=-1)  # (H, W)
    floor_map = floor_base + floor_obs_scale / (1.0 + n_obs)     # (H, W)
    return floor_map[..., np.newaxis]  # (H, W, 1)


def build_predictions_pooled(
    alphas: list[np.ndarray],
    observations: dict[int, np.ndarray],
    terrains: list[np.ndarray],
    settlements_list: list[list[dict]],
    extra_obs: dict[int, np.ndarray] | None = None,
    pool_weight: float = 1.0,
    temperature: float = 1.0,
    floor: float | None = None,
    floor_base: float = 0.002,
    floor_obs_scale: float = 0.02,
) -> list[np.ndarray]:
    """Fuse priors with both cell-level and pooled feature-bin observations.

    α_post(u) = α_prior(u) + obs(u) + pool_weight · pooled(key(u))

    Works internally in 8-state space (preserving ocean/plains/empty
    distinction), then collapses to 6-class for submission.

    Supports temperature scaling (T<1 sharpens, T>1 softens) and dynamic
    per-cell probability floors based on observation evidence.

    Args:
        alphas:           list of (H, W, 8) Dirichlet prior pseudo-counts per seed.
        observations:     dict[seed_idx] -> (H, W, 8) observation counts.
        terrains:         list of (H, W) terrain grids per seed.
        settlements_list: list of settlement lists per seed.
        extra_obs:        optional dict[seed_idx] -> (H, W, 8) additional counts.
        pool_weight:      scaling factor for pooled evidence (1.0 = full pooling).
        temperature:      posterior temperature (1.0 = no change, <1 = sharpen).
        floor:            if set, use this fixed floor (overrides dynamic floor).
        floor_base:       dynamic floor: minimum floor for well-observed cells.
        floor_obs_scale:  dynamic floor: extra floor for unobserved cells.

    Returns:
        list of (H, W, 6) normalized probability tensors per seed (collapsed).
    """
    # Merge all observations for pooling
    all_obs = {}
    for seed_idx in observations:
        merged = observations[seed_idx].copy()
        if extra_obs and seed_idx in extra_obs:
            merged += extra_obs[seed_idx]
        all_obs[seed_idx] = merged

    pooled = pool_observations_by_feature(
        all_obs, terrains, settlements_list,
    )

    predictions = []
    for seed_idx, alpha_prior in enumerate(alphas):
        terrain = terrains[seed_idx]
        setts = settlements_list[seed_idx]
        tc_map, db_map, _, co_map, den_map = compute_cell_features(
            terrain, setts,
        )
        H, W = terrain.shape
        alpha_post = alpha_prior.copy()

        # Add cell-level observations
        if seed_idx in observations:
            alpha_post += observations[seed_idx]
        if extra_obs and seed_idx in extra_obs:
            alpha_post += extra_obs[seed_idx]

        # Add pooled observations by feature key
        if pool_weight > 0:
            for y in range(H):
                for x in range(W):
                    key = (int(tc_map[y, x]), int(db_map[y, x]),
                           int(co_map[y, x]), int(den_map[y, x]))
                    if key in pooled:
                        cell_obs = all_obs.get(seed_idx, np.zeros_like(alpha_prior))
                        pooled_other = pooled[key] - cell_obs[y, x]
                        pooled_other = np.maximum(pooled_other, 0.0)
                        alpha_post[y, x] += pool_weight * pooled_other

        # Temperature scaling: α^(1/T)
        if temperature != 1.0:
            alpha_post = np.power(np.maximum(alpha_post, 1e-12), 1.0 / temperature)

        # Normalize in 8-state, then collapse to 6-class for submission
        q8 = alpha_post / alpha_post.sum(axis=-1, keepdims=True)
        q = collapse_8state_to_6class(q8)

        # Apply floor: fixed or dynamic
        if floor is not None:
            q = apply_floor_and_normalize(q, floor=floor)
        else:
            floor_per_cell = dynamic_floor(
                alpha_post, alpha_prior, floor_base, floor_obs_scale,
            )
            q = np.maximum(q, floor_per_cell)
            q = q / q.sum(axis=-1, keepdims=True)

        predictions.append(q)
    return predictions
