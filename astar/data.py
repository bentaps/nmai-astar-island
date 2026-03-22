"""
Data I/O helpers: loading round data, saving observations, fetching ground truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "rounds"
LOOKUP_PATH = Path(__file__).parent.parent / "data" / "dirichlet_lookup.json"


def parse_initial_states(round_data: dict) -> tuple[list, list]:
    """Parse round details into terrain grids and settlement lists.

    Args:
        round_data: dict from GET /rounds/{id}.

    Returns:
        terrains:         list of (H, W) int arrays per seed.
        settlements_list: list of settlement dicts per seed.
    """
    terrains = []
    settlements_list = []
    for state in round_data["initial_states"]:
        terrains.append(np.array(state["grid"], dtype=int))
        settlements_list.append(state.get("settlements", []))
    return terrains, settlements_list


def save_round_data(
    round_id: str,
    round_data: dict,
    observations: dict,
    predictions: list,
    extra_obs: dict | None = None,
) -> None:
    """Save all round data for future cross-round learning.

    Writes to data/rounds/{round_id}/:
        round_data.json
        observations_seed{i}.npy
        extra_obs_seed{i}.npy  (if extra_obs provided)
        prediction_seed{i}.npy

    Args:
        round_id:     round UUID.
        round_data:   dict from GET /rounds/{id}.
        observations: dict[seed_idx] -> (H, W, 6) count array.
        predictions:  list of (H, W, 6) probability tensors per seed.
        extra_obs:    optional dict[seed_idx] -> (H, W, 6) extra counts.
    """
    round_dir = DATA_DIR / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    (round_dir / "round_data.json").write_text(json.dumps(round_data, indent=2))

    for seed_idx, obs in observations.items():
        np.save(round_dir / f"observations_seed{seed_idx}.npy", obs)

    if extra_obs:
        for seed_idx, obs in extra_obs.items():
            np.save(round_dir / f"extra_obs_seed{seed_idx}.npy", obs)

    for seed_idx, pred in enumerate(predictions):
        np.save(round_dir / f"prediction_seed{seed_idx}.npy", pred)

    print(f"Round data saved to {round_dir}")


def save_query_log(round_id: str, settlement_log: list[dict]) -> None:
    """Save enriched query log as JSON for future model training.

    Each entry contains seed_idx, viewport, query_index, settlements, raw_grid.
    Also extracts and saves settlement_records.json (per-seed settlement data
    for offline replay via SimulatedSession).
    """
    round_dir = DATA_DIR / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    path = round_dir / "query_log.json"
    path.write_text(json.dumps(settlement_log, indent=2))
    print(f"  Query log saved to {path} ({len(settlement_log)} queries)")

    # Extract settlement records per seed (use the query with most settlements
    # for each seed, since different viewports see different subsets)
    sett_by_seed: dict[int, list[dict]] = {}
    for entry in settlement_log:
        if not isinstance(entry, dict):
            continue
        seed_idx = entry.get("seed_idx")
        settlements = entry.get("settlements", [])
        if seed_idx is None or not settlements:
            continue
        # Merge: accumulate unique settlements by (x, y) position
        if seed_idx not in sett_by_seed:
            sett_by_seed[seed_idx] = {}
        for s in settlements:
            key = (s.get("x", -1), s.get("y", -1))
            sett_by_seed[seed_idx][key] = s

    if sett_by_seed:
        records = {str(k): list(v.values()) for k, v in sett_by_seed.items()}
        sett_path = round_dir / "settlement_records.json"
        sett_path.write_text(json.dumps(records, indent=2))
        total = sum(len(v) for v in records.values())
        print(f"  Settlement records saved to {sett_path} "
              f"({total} settlements across {len(records)} seeds)")


def fetch_and_save_ground_truth(
    session,
    round_id: str,
    num_seeds: int,
    get_analysis_fn,
) -> None:
    """Fetch ground truth from the analysis endpoint (only works post-round).

    Args:
        session:         RateLimitedSession.
        round_id:        round UUID.
        num_seeds:       number of seeds.
        get_analysis_fn: callable(session, round_id, seed_idx) -> dict.
    """
    round_dir = DATA_DIR / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    for seed_idx in range(num_seeds):
        try:
            analysis = get_analysis_fn(session, round_id, seed_idx)
            (round_dir / f"analysis_seed{seed_idx}.json").write_text(
                json.dumps(analysis, indent=2)
            )
            print(f"  Ground truth saved for seed {seed_idx}")
        except requests.HTTPError as e:
            print(f"  Could not fetch analysis for seed {seed_idx}: {e}")
