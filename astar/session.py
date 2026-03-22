"""
SimulatedSession — offline fake API for method development and comparison.

Use this for all experimentation. No network calls, no budget consumed.
It has exactly the same .simulate() interface as LiveSession (astar.api),
so any inference code that works here will work against the real API.

Typical use
-----------
    from astar.session import SimulatedSession

    ground_truth = [...]          # list of (H, W, 6) arrays per seed
    sim = SimulatedSession(ground_truth, budget=50)
    result = sim.simulate(seed_idx=0, x=0, y=0, w=15, h=15)
    # result has the same schema as the real /simulate endpoint
"""

from __future__ import annotations

import numpy as np

from .features import NUM_RAW_CODES, IDX_TO_RAW_CODE


class SimulatedSession:
    """Mimics the real API using historical ground truth distributions.

    Each call to .simulate() draws one independent Monte Carlo sample
    per cell from the ground truth distribution — exactly what the real
    API does inside the competition server.

    Args:
        ground_truth:       list of (H, W, 8) float probability arrays (8-state),
                            one per seed.
        budget:             maximum queries allowed (default 50).
        rng_seed:           random seed for reproducibility.
        settlement_records: optional dict[seed_idx_str] -> list[settlement_dict].
                            If provided, simulate() returns settlement data like
                            the real API. Load from settlement_records.json.
    """

    def __init__(
        self,
        ground_truth: list[np.ndarray],
        budget: int = 50,
        rng_seed: int = 42,
        settlement_records: dict[str, list[dict]] | None = None,
    ):
        self.gt = ground_truth
        self.budget_max = budget
        self.queries_used = 0
        self.queries_max = budget
        self.rng = np.random.default_rng(rng_seed)
        self.query_log: list[tuple] = []   # (seed_idx, x, y, w, h)
        self.settlement_records = settlement_records or {}

    def simulate(
        self,
        seed_idx: int,
        x: int, y: int,
        w: int = 15, h: int = 15,
    ) -> dict:
        """Sample one stochastic viewport from the GT distribution.

        Returns 8-state terrain codes matching the live API format:
            grid:         list[list[int]]  — sampled raw terrain code per cell
            viewport:     {x, y, w, h}
            settlements:  list[dict]       — settlement payloads (if available)
            queries_used: int
            queries_max:  int
        """
        if self.queries_used >= self.budget_max:
            raise RuntimeError(
                f"SimulatedSession budget exhausted ({self.budget_max} queries)"
            )

        H, W = self.gt[seed_idx].shape[:2]
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = min(w, W - x)
        h = min(h, H - y)

        n_states = self.gt[seed_idx].shape[-1]
        gt_vp = self.gt[seed_idx][y:y+h, x:x+w]   # (h, w, 8) or (h, w, 6)

        grid = []
        for row_idx in range(h):
            row = []
            for col_idx in range(w):
                p = gt_vp[row_idx, col_idx].copy()
                p = np.maximum(p, 0.0)
                total = p.sum()
                if total > 0:
                    p /= total
                else:
                    p = np.ones(n_states) / n_states
                state_idx = int(self.rng.choice(n_states, p=p))
                # Convert 8-state index to raw terrain code (matching live API)
                if n_states == NUM_RAW_CODES:
                    code = IDX_TO_RAW_CODE[state_idx]
                else:
                    code = state_idx  # legacy 6-class: code = class index
                row.append(code)
            grid.append(row)

        self.queries_used += 1
        self.query_log.append((seed_idx, x, y, w, h))

        # Return settlement records filtered to viewport (matching real API)
        all_setts = self.settlement_records.get(str(seed_idx), [])
        settlements = [
            s for s in all_setts
            if x <= s.get("x", -1) < x + w and y <= s.get("y", -1) < y + h
        ]

        return {
            "grid":         grid,
            "viewport":     {"x": x, "y": y, "w": w, "h": h},
            "settlements":  settlements,
            "queries_used": self.queries_used,
            "queries_max":  self.budget_max,
        }

    @property
    def budget_remaining(self) -> int:
        return self.budget_max - self.queries_used

    def reset(self, rng_seed: int | None = None) -> "SimulatedSession":
        """Reset query count and optionally re-seed the RNG."""
        self.queries_used = 0
        self.query_log = []
        if rng_seed is not None:
            self.rng = np.random.default_rng(rng_seed)
        return self
