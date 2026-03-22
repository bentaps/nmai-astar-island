"""
Spatial feature extraction for Astar Island terrain grids.

Feature vector per cell:
  k(u) = (terrain_code, dist_bin, is_coastal, density_bin)

  terrain_code  — 8-code initial terrain (ocean, plains, forest, …)
  dist_bin      — binned Manhattan distance to nearest settlement
  is_coastal    — 1 if cell is 4-connected adjacent to ocean
  density_bin   — binned count of settlements within radius r

These four features capture the main mechanical drivers:
  physical substrate, settlement influence, port/trade access, and
  whether a cell sits inside a dense urban cluster or a lone frontier.
"""

import numpy as np
from collections import deque

# Internal terrain codes → 6 submission classes
# Codes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains
TERRAIN_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 0, 11: 0}
NUM_CLASSES = 6

# Raw 8-state terrain code → internal index (preserves ocean/plains/empty distinction)
RAW_CODE_TO_IDX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6, 11: 7}
NUM_RAW_CODES = 8

# Collapse 8-state index back to 6 submission classes
RAW_IDX_TO_CLASS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 0, 7: 0}

# Reverse mapping: 8-state index → raw API terrain code
IDX_TO_RAW_CODE = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 10, 7: 11}


def expand_6class_to_8state(
    Y: np.ndarray,
    terrain: np.ndarray,
) -> np.ndarray:
    """Expand (H, W, 6) ground truth to (H, W, 8) using initial terrain.

    Classes 1-5 map 1:1 to states 1-5. Class 0 is split:
      - Ocean cells (terrain=10): class 0 probability → state 6 (ocean)
      - All other cells:          class 0 probability → state 7 (plains)
    State 0 (empty) gets zero probability since code 0 is never observed.

    Also works with batched input: (N, H, W, 6) + (N, H, W) → (N, H, W, 8).
    """
    if Y.ndim == 4:
        # Batched: (N, H, W, 6) + (N, H, W)
        N = Y.shape[0]
        out = np.zeros((*Y.shape[:-1], NUM_RAW_CODES), dtype=Y.dtype)
        for i in range(N):
            out[i] = expand_6class_to_8state(Y[i], terrain[i])
        return out

    # Single: (H, W, 6) + (H, W)
    out = np.zeros((*Y.shape[:-1], NUM_RAW_CODES), dtype=Y.dtype)
    out[..., 1:6] = Y[..., 1:6]  # classes 1-5 → states 1-5
    ocean_mask = terrain == 10
    out[ocean_mask, 6] = Y[ocean_mask, 0]      # ocean cells: class 0 → state 6
    out[~ocean_mask, 7] = Y[~ocean_mask, 0]    # non-ocean: class 0 → state 7
    return out


def raw_grid_to_8state(grid: list[list[int]]) -> np.ndarray:
    """Convert raw API grid to (h, w) array of 8-state indices."""
    arr = np.array(grid, dtype=int)
    return np.vectorize(lambda c: RAW_CODE_TO_IDX.get(c, 0))(arr)


def collapse_8state_to_6class(raw_obs: np.ndarray) -> np.ndarray:
    """Collapse (H, W, 8) observation counts to (H, W, 6) for submission."""
    out = np.zeros((*raw_obs.shape[:-1], NUM_CLASSES), dtype=raw_obs.dtype)
    for raw_idx, cls in RAW_IDX_TO_CLASS.items():
        out[..., cls] += raw_obs[..., raw_idx]
    return out

# Distance-to-settlement bin edges.
DEFAULT_DIST_BINS = np.array([0, 1, 2, 3, 4, 6, 8, 12, 20, 9999])

# Settlement density bins — count of settlements within DEFAULT_DENSITY_RADIUS.
DEFAULT_DENSITY_RADIUS = 6
DEFAULT_DENSITY_BINS   = np.array([0, 1, 2, 4, 9999])   # → 4 bins: none / 1 / 2-3 / 4+


def distance_to_nearest_settlement(terrain: np.ndarray, settlements: list) -> np.ndarray:
    """Manhattan distance from each cell to the nearest settlement (BFS).

    Args:
        terrain:     (H, W) terrain code grid (used only for shape).
        settlements: list of dicts with 'x' and 'y' keys.

    Returns:
        (H, W) int32 distances; cells with no settlement get 9999.
    """
    H, W = terrain.shape
    dist = np.full((H, W), 9999, dtype=np.int32)
    queue = deque()
    for s in settlements:
        sx, sy = s["x"], s["y"]
        if 0 <= sy < H and 0 <= sx < W:
            dist[sy, sx] = 0
            queue.append((sy, sx))
    while queue:
        y, x = queue.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                nd = dist[y, x] + 1
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    queue.append((ny, nx))
    return dist


def bin_distances(dist: np.ndarray, bins: np.ndarray = DEFAULT_DIST_BINS) -> np.ndarray:
    """Digitize distances into bin indices (0-indexed)."""
    return np.digitize(dist, bins) - 1


def is_coastal(terrain: np.ndarray) -> np.ndarray:
    """Boolean mask: True for land cells 4-connected adjacent to ocean (code 10).

    Ocean cells themselves are False.
    """
    ocean = terrain == 10
    coastal = np.zeros_like(ocean)
    coastal[1:,  :] |= ocean[:-1, :]   # cell above is ocean
    coastal[:-1, :] |= ocean[1:,  :]   # cell below is ocean
    coastal[:,  1:] |= ocean[:, :-1]   # cell to the left is ocean
    coastal[:, :-1] |= ocean[:,  1:]   # cell to the right is ocean
    return coastal & ~ocean


def settlement_density(
    terrain: np.ndarray,
    settlements: list,
    radius: int = DEFAULT_DENSITY_RADIUS,
) -> np.ndarray:
    """Count settlements within Manhattan distance `radius` of each cell.

    Args:
        terrain:     (H, W) terrain grid (used for shape).
        settlements: list of settlement dicts with 'x', 'y' keys.
        radius:      Manhattan radius.

    Returns:
        (H, W) int32 count array.
    """
    H, W = terrain.shape
    density = np.zeros((H, W), dtype=np.int32)
    for s in settlements:
        sx, sy = s["x"], s["y"]
        for dy in range(-radius, radius + 1):
            for dx in range(-(radius - abs(dy)), radius - abs(dy) + 1):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    density[ny, nx] += 1
    return density


def compute_cell_features(
    terrain: np.ndarray,
    settlements: list,
    dist_bins:    np.ndarray = DEFAULT_DIST_BINS,
    density_bins: np.ndarray = DEFAULT_DENSITY_BINS,
    density_radius: int      = DEFAULT_DENSITY_RADIUS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all per-cell features for lookup-table indexing.

    Returns:
        terrain_codes:    (H, W) terrain code.
        dist_bin_indices: (H, W) distance-to-settlement bin (0-indexed).
        raw_distances:    (H, W) raw Manhattan distances.
        coastal_mask:     (H, W) bool — True for cells adjacent to ocean.
        density_bin_indices: (H, W) settlement-density bin (0-indexed).
    """
    raw_distances   = distance_to_nearest_settlement(terrain, settlements)
    dist_bin_idx    = bin_distances(raw_distances, dist_bins)
    coastal_mask    = is_coastal(terrain)
    density_counts  = settlement_density(terrain, settlements, radius=density_radius)
    density_bin_idx = np.digitize(density_counts, density_bins) - 1
    return terrain.copy(), dist_bin_idx, raw_distances, coastal_mask, density_bin_idx
