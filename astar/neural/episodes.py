"""
Episode generator: turn each historical round into many training tasks.

Each episode:
  - Starts from initial terrain + settlements for all seeds of one round
  - Samples a synthetic query history (random subset of viewports)
  - Simulates stochastic observations from ground truth
  - Computes v3 baseline posterior
  - Target: full 6-class ground truth tensor

This creates thousands of (context, target) pairs from ~16 rounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data import parse_initial_states
from ..dirichlet import DirichletLookup
from ..evaluation import load_dataset
from ..features import (
    NUM_RAW_CODES,
    NUM_CLASSES,
    RAW_CODE_TO_IDX,
    IDX_TO_RAW_CODE,
    collapse_8state_to_6class,
    compute_cell_features,
    is_coastal,
    distance_to_nearest_settlement,
    settlement_density,
)
from ..inference import (
    build_predictions_pooled,
    dynamic_coverage_viewports,
    pool_observations_by_feature,
)
from ..scoring import apply_floor_and_normalize
from ..session import SimulatedSession


def build_feature_planes(
    terrain: np.ndarray,
    settlements: list[dict],
    obs_8: np.ndarray,
    v3_posterior_6: np.ndarray,
) -> np.ndarray:
    """Build per-cell feature tensor for the neural model.

    Returns (H, W, C_feat) float32 tensor with channels:
      [0:8]   terrain one-hot (8 raw codes)
      [8:16]  observation counts (8-state)
      [16]    total observation count per cell
      [17:23] v3 posterior 6-class probabilities
      [23]    v3 posterior entropy
      [24]    is_coastal flag
      [25]    distance to nearest settlement (normalized)
      [26]    settlement density (normalized)
    """
    H, W = terrain.shape

    # Terrain one-hot (8 channels)
    terrain_oh = np.zeros((H, W, NUM_RAW_CODES), dtype=np.float32)
    code_to_idx = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6, 11: 7}
    for y in range(H):
        for x in range(W):
            idx = code_to_idx.get(int(terrain[y, x]), 0)
            terrain_oh[y, x, idx] = 1.0

    # Observation counts (8-state) + total
    obs_total = obs_8.sum(axis=-1, keepdims=True)  # (H, W, 1)

    # V3 posterior + entropy
    v3_safe = np.clip(v3_posterior_6, 1e-12, 1.0)
    v3_entropy = -np.sum(v3_safe * np.log(v3_safe), axis=-1, keepdims=True)

    # Spatial features
    coastal = is_coastal(terrain).astype(np.float32)[..., np.newaxis]
    dist = distance_to_nearest_settlement(terrain, settlements).astype(np.float32)
    dist_norm = np.minimum(dist / 20.0, 1.0)[..., np.newaxis]
    density = settlement_density(terrain, settlements).astype(np.float32)
    density_norm = np.minimum(density / 6.0, 1.0)[..., np.newaxis]

    return np.concatenate([
        terrain_oh,           # 8
        obs_8.astype(np.float32),  # 8
        obs_total.astype(np.float32),  # 1
        v3_posterior_6.astype(np.float32),  # 6
        v3_entropy.astype(np.float32),  # 1
        coastal,              # 1
        dist_norm,            # 1
        density_norm,         # 1
    ], axis=-1)  # total: 27 channels


def build_round_latent_features(
    all_features: list[np.ndarray],
    all_obs: list[np.ndarray],
) -> np.ndarray:
    """Pool features across seeds to build a round-level context vector.

    Args:
        all_features: list of (H, W, C_feat) per seed
        all_obs: list of (H, W, 8) observation counts per seed

    Returns:
        (D_round,) vector summarizing the round.
    """
    # Simple pooling: mean and max of key statistics across seeds
    stats = []
    for feat in all_features:
        # Mean v3 entropy across observed cells
        obs_mask = feat[..., 16] > 0  # total obs > 0
        if obs_mask.any():
            stats.append(feat[obs_mask, 23].mean())  # mean entropy of observed
            stats.append(feat[obs_mask, 23].max())    # max entropy
            stats.append(float(obs_mask.mean()))       # observation coverage
        else:
            stats.extend([0.0, 0.0, 0.0])

    # Pad/truncate to fixed size (5 seeds × 3 stats = 15, + overall stats)
    while len(stats) < 15:
        stats.append(0.0)
    stats = stats[:15]

    # Add cross-seed observation agreement
    if len(all_obs) >= 2:
        # Mean pairwise correlation of observation patterns
        obs_flat = [o.reshape(-1, NUM_RAW_CODES).sum(axis=-1) for o in all_obs]
        corrs = []
        for i in range(len(obs_flat)):
            for j in range(i + 1, len(obs_flat)):
                c = np.corrcoef(obs_flat[i], obs_flat[j])[0, 1]
                corrs.append(c if np.isfinite(c) else 0.0)
        stats.append(float(np.mean(corrs)) if corrs else 0.0)
    else:
        stats.append(0.0)

    return np.array(stats, dtype=np.float32)  # (16,)


class EpisodeGenerator:
    """Generate training episodes from historical rounds.

    Each episode simulates a partial-query history for one round and
    computes v3 baseline predictions, producing (features, target) pairs.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        pool_weight: float = 1.581,
        temperature: float = 1.041,
        floor_base: float = 0.0002143,
        floor_obs_scale: float = 0.008436,
    ):
        self.data_dir = Path(data_dir)
        self.pool_weight = pool_weight
        self.temperature = temperature
        self.floor_base = floor_base
        self.floor_obs_scale = floor_obs_scale

        self.dataset = load_dataset(self.data_dir)
        self._prepare_rounds()

    def _prepare_rounds(self) -> None:
        """Organize dataset by round for episode generation."""
        meta = self.dataset["meta"]
        self.rounds: dict[int, dict] = {}

        for round_num in self.dataset["round_numbers"]:
            idx = sorted(
                [i for i, m in enumerate(meta) if m["round_num"] == round_num],
                key=lambda i: meta[i]["seed_idx"],
            )
            if not idx:
                continue

            round_id = meta[idx[0]]["round_id"]
            rd_path = self.data_dir / "rounds" / round_id / "round_data.json"
            if not rd_path.exists():
                continue

            round_data = json.loads(rd_path.read_text())
            terrains, setts_list = parse_initial_states(round_data)

            # Load settlement records if available
            sett_path = self.data_dir / "rounds" / round_id / "settlement_records.json"
            sett_records = (json.loads(sett_path.read_text())
                           if sett_path.exists() else None)

            self.rounds[round_num] = {
                "idx": idx,
                "round_id": round_id,
                "terrains": terrains,
                "setts_list": setts_list,
                "sett_records": sett_records,
                "H": terrains[0].shape[0],
                "W": terrains[0].shape[1],
            }

    def generate_episode(
        self,
        round_num: int,
        n_queries: int = 50,
        rng_seed: int = 42,
        train_rounds: list[int] | None = None,
    ) -> dict | None:
        """Generate one training episode for a round.

        Args:
            round_num: which round to simulate.
            n_queries: query budget for this episode.
            rng_seed: RNG seed for stochastic simulation.
            train_rounds: which rounds to use for prior fitting.
                         If None, uses all rounds < round_num.

        Returns:
            dict with keys:
                features_per_seed: list of (H, W, 27) feature tensors
                round_latent: (16,) round-level context vector
                v3_preds_per_seed: list of (H, W, 6) v3 posterior predictions
                targets_per_seed: list of (H, W, 6) ground truth tensors
                terrains: list of (H, W) terrain grids
                round_num: int
        """
        if round_num not in self.rounds:
            return None

        rinfo = self.rounds[round_num]
        meta = self.dataset["meta"]
        X = self.dataset["X"]
        Y = self.dataset["Y"]       # 8-state
        Y6 = self.dataset["Y6"]     # 6-class

        # Determine training set
        if train_rounds is None:
            train_idx = [i for i, m in enumerate(meta) if m["round_num"] < round_num]
        else:
            train_idx = [i for i, m in enumerate(meta) if m["round_num"] in train_rounds]

        if not train_idx:
            return None

        # Fit prior
        setts = self.dataset["settlements"]
        lookup = DirichletLookup()
        lookup.fit(X[train_idx], Y[train_idx], [setts[i] for i in train_idx])

        test_idx = rinfo["idx"]
        terrains = rinfo["terrains"]
        setts_list = rinfo["setts_list"]
        num_seeds = len(test_idx)
        H, W = rinfo["H"], rinfo["W"]

        # Ground truth
        gt_8 = [Y[i] for i in test_idx]
        gt_6 = [Y6[i] for i in test_idx]

        # Build priors
        alphas = [lookup.build_prior(terrains[s], setts_list[s])
                  for s in range(num_seeds)]

        # Simulate queries
        sim = SimulatedSession(
            gt_8, budget=n_queries, rng_seed=rng_seed,
            settlement_records=rinfo["sett_records"],
        )

        # Use the same uniform strategy as backtest
        resample_budget = max(5, n_queries // 10)
        vps_per_seed = (n_queries - resample_budget) // num_seeds

        viewports_per_seed = [
            dynamic_coverage_viewports(
                terrains[s], setts_list[s],
                max_viewports=vps_per_seed,
                max_settlement_dist=8,
            )
            for s in range(num_seeds)
        ]

        # Run queries
        observations: dict[int, np.ndarray] = {}
        for seed_idx in range(num_seeds):
            obs = np.zeros((H, W, NUM_RAW_CODES), dtype=float)
            for vx, vy, vw, vh in viewports_per_seed[seed_idx]:
                result = sim.simulate(seed_idx, vx, vy, vw, vh)
                vp = result["viewport"]
                for ri, row in enumerate(result["grid"]):
                    for ci, code in enumerate(row):
                        gy, gx = vp["y"] + ri, vp["x"] + ci
                        obs[gy, gx, RAW_CODE_TO_IDX.get(code, 0)] += 1.0
            observations[seed_idx] = obs

        # Resampling
        remaining = sim.budget_remaining
        alphas_post = [alphas[s] + observations[s] for s in range(num_seeds)]

        extra_obs: dict[int, np.ndarray] = {
            i: np.zeros((H, W, NUM_RAW_CODES), dtype=float)
            for i in range(num_seeds)
        }
        if remaining > 0:
            from ..inference import rank_windows_by_entropy
            candidates = []
            for seed_idx in range(num_seeds):
                ranked = rank_windows_by_entropy(alphas_post[seed_idx])
                for score, x, y, w, h in ranked[:3]:
                    candidates.append((score, seed_idx, x, y, w, h))
            candidates.sort(reverse=True)

            for score, seed_idx, vx, vy, vw, vh in candidates[:remaining]:
                result = sim.simulate(seed_idx, vx, vy, vw, vh)
                vp = result["viewport"]
                for ri, row in enumerate(result["grid"]):
                    for ci, code in enumerate(row):
                        gy, gx = vp["y"] + ri, vp["x"] + ci
                        extra_obs[seed_idx][gy, gx, RAW_CODE_TO_IDX.get(code, 0)] += 1.0

        # Compute v3 predictions
        v3_preds = build_predictions_pooled(
            alphas, observations, terrains, setts_list,
            extra_obs=extra_obs,
            pool_weight=self.pool_weight,
            temperature=self.temperature,
            floor_base=self.floor_base,
            floor_obs_scale=self.floor_obs_scale,
        )

        # Build feature planes and round latent
        total_obs = {s: observations[s] + extra_obs[s] for s in range(num_seeds)}
        features_per_seed = []
        for s in range(num_seeds):
            feat = build_feature_planes(
                terrains[s], setts_list[s], total_obs[s], v3_preds[s],
            )
            features_per_seed.append(feat)

        round_latent = build_round_latent_features(
            features_per_seed, [total_obs[s] for s in range(num_seeds)],
        )

        return {
            "features_per_seed": features_per_seed,   # list of (H, W, 27)
            "round_latent": round_latent,              # (16,)
            "v3_preds_per_seed": v3_preds,             # list of (H, W, 6)
            "targets_per_seed": gt_6,                  # list of (H, W, 6)
            "terrains": terrains,                      # list of (H, W)
            "round_num": round_num,
            "num_seeds": num_seeds,
        }

    def generate_batch(
        self,
        test_rounds: list[int] | None = None,
        episodes_per_round: int = 10,
        budget: int | list[int] = 50,
        base_seed: int = 0,
        prior_rounds: list[int] | None = None,
    ) -> list[dict]:
        """Generate a batch of training episodes across rounds.

        Args:
            test_rounds: which rounds to generate episodes for.
                        If None, uses rounds 6+.
            episodes_per_round: number of RNG seeds per round.
            budget: query budget per episode. If a list, cycles through
                   different budgets for diversity.
            base_seed: starting RNG seed.
            prior_rounds: rounds to use for Dirichlet prior fitting.
                         If None, uses all rounds except the test round.
                         This ensures v3 baseline quality matches test-time.

        Returns:
            List of episode dicts.
        """
        if test_rounds is None:
            test_rounds = [r for r in self.rounds if r >= 6]

        budgets = budget if isinstance(budget, list) else [budget]
        all_available = sorted(self.rounds.keys())

        episodes = []
        for round_num in sorted(test_rounds):
            # Use only rounds strictly earlier than the test round (chronology-faithful)
            if prior_rounds is not None:
                tr = [r for r in prior_rounds if r < round_num]
            else:
                tr = [r for r in all_available if r < round_num]

            for ep in range(episodes_per_round):
                b = budgets[ep % len(budgets)]
                rng_seed = base_seed + round_num * 1000 + ep
                episode = self.generate_episode(
                    round_num, n_queries=b, rng_seed=rng_seed,
                    train_rounds=tr,
                )
                if episode is not None:
                    episodes.append(episode)

        return episodes
