"""
Data-driven Dirichlet prior via entropy-weighted method of moments.

Feature key per cell:
    k(u) = (terrain_code, dist_bin, is_coastal, density_bin)

Fitting procedure
-----------------
1.  Collect ground-truth distributions per feature bin.

2.  Compute the bin mean using an *entropy-weighted* average:

        μ_k = Σ_u H(p_u) p_u  /  Σ_u H(p_u)

    This aligns the mean with the competition scoring rule, which weights
    uncertain cells more heavily than trivially static ones.

3.  Apply *hierarchical shrinkage* on the mean before estimating α:

    The fine bin mean is shrunk toward its parent (coarser) bin mean.
    Parent bins drop features in order: density → dist → coastal.
    Shrinkage weight:  λ = n / (n + τ),  default τ = 10.

    This avoids the hard "< min_samples → discard" switch and gives a
    smooth transition that exploits all available data.

4.  Choose concentration κ via *shrinkage toward terrain-group defaults*:

        κ_k = λ_κ · κ_MoM  +  (1 − λ_κ) · κ_group(terrain_code, coastal)
        λ_κ = n / (n + n_reg),   default n_reg = 50

    The MoM formula  κ = median_c[ μ_c(1−μ_c)/Var[p_c] − 1 ]  is correct
    under the Dirichlet model, but empirical variance across a bin reflects
    bin heterogeneity as much as true uncertainty.  Shrinking toward stable
    terrain-group defaults prevents wild values when bins are small or
    heterogeneous.

5.  Final pseudo-counts:  α_k = κ_k · μ_k^shrunk + ε

Lookup fallback hierarchy (at inference time, for unseen feature combos):
    (tc, db, co, den)  →  (tc, co)  →  global
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import (
    NUM_CLASSES,
    NUM_RAW_CODES,
    compute_cell_features,
    DEFAULT_DIST_BINS,
    DEFAULT_DENSITY_BINS,
    DEFAULT_DENSITY_RADIUS,
)

# ---------------------------------------------------------------------------
# Terrain-group concentration defaults
# Static terrain gets a strong prior; dynamic / coastal gets a weak one.
# ---------------------------------------------------------------------------
_TERRAIN_KAPPA: dict[int, float] = {
    10: 200.0,  # Ocean     — never changes
    5:  200.0,  # Mountain  — never changes
    4:   15.0,  # Forest    — mostly static
    11:   5.0,  # Plains    — dynamic frontier
    0:    5.0,  # Empty     — dynamic
    1:    4.0,  # Settlement — dynamic
    3:    4.0,  # Ruin       — dynamic
    2:    3.0,  # Port       — very dynamic (coastal)
}
_COASTAL_KAPPA_FACTOR = 0.6   # multiply non-static coastal cells' kappa


def _terrain_group_kappa(terrain_code: int, coastal: int) -> float:
    """Default concentration for (terrain, coastal) — stable terrain-group prior."""
    kappa = _TERRAIN_KAPPA.get(terrain_code, 5.0)
    if coastal == 1 and terrain_code not in (10, 5):
        kappa *= _COASTAL_KAPPA_FACTOR
    return kappa


def _entropy_weighted_mean(distributions: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Entropy-weighted mean of a batch of distributions.

    Weights each p_u by H(p_u) so that uncertain (dynamic) cells
    drive the mean more than trivially static cells.

    Args:
        distributions: (n, C) probability tensors.

    Returns:
        (C,) weighted mean.
    """
    p = np.clip(distributions, eps, 1.0)
    h = -np.sum(p * np.log(p), axis=1)   # (n,)
    total_h = h.sum()
    if total_h < eps:
        return distributions.mean(axis=0)
    return (distributions * h[:, None]).sum(axis=0) / total_h


class DirichletLookup:
    """Fitted Dirichlet prior lookup table.

    Feature key: (terrain_code, dist_bin, is_coastal, density_bin)

    Attributes:
        table:            dict (int,int,int,int) → np.ndarray (NUM_CLASSES,)
        terrain_fallback: dict (int,int) → np.ndarray  — (tc, coastal)
        global_fallback:  np.ndarray (NUM_CLASSES,)
        tau:              mean shrinkage regularisation strength
        n_reg:            kappa shrinkage regularisation sample count
    """

    def __init__(
        self,
        kappa_min:      float = 5.678,
        kappa_max:      float = 200.0,
        epsilon:        float = 0.03423,
        tau:            float = 9.775,
        n_reg:          float = 1.171,
        dist_bins:      np.ndarray = DEFAULT_DIST_BINS,
        density_bins:   np.ndarray = DEFAULT_DENSITY_BINS,
        density_radius: int        = DEFAULT_DENSITY_RADIUS,
    ):
        self.kappa_min      = kappa_min
        self.kappa_max      = kappa_max
        self.epsilon        = epsilon
        self.tau            = tau
        self.n_reg          = n_reg
        self.dist_bins      = dist_bins
        self.density_bins   = density_bins
        self.density_radius = density_radius

        self.table:            dict[tuple, np.ndarray] = {}
        self.terrain_fallback: dict[tuple, np.ndarray] = {}
        self.global_fallback = np.full(NUM_RAW_CODES, 1.0)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        settlements_per_sample: list[list[dict]],
    ) -> "DirichletLookup":
        """Fit the lookup table from training data.

        Args:
            X: (N, H, W) initial terrain codes.
            Y: (N, H, W, C) ground truth probability tensors.
            settlements_per_sample: length-N list of settlement lists.

        Returns:
            self (for chaining).
        """
        # Collect distributions at three hierarchy levels
        fine_data:    dict[tuple, list] = defaultdict(list)  # (tc,db,co,den)
        mid_data:     dict[tuple, list] = defaultdict(list)  # (tc,db,co)
        coarse_data:  dict[tuple, list] = defaultdict(list)  # (tc,co)

        for i in range(X.shape[0]):
            tc_map, db_map, _, co_map, den_map = compute_cell_features(
                X[i], settlements_per_sample[i], self.dist_bins,
                self.density_bins, self.density_radius,
            )
            H, W = X[i].shape
            for y in range(H):
                for x in range(W):
                    tc  = int(tc_map[y, x])
                    db  = int(db_map[y, x])
                    co  = int(co_map[y, x])
                    den = int(den_map[y, x])
                    gt  = Y[i, y, x]
                    fine_data  [(tc, db, co, den)].append(gt)
                    mid_data   [(tc, db, co)      ].append(gt)
                    coarse_data[(tc, co)           ].append(gt)

        # --- Compute entropy-weighted means at each level ----------------
        def ew_mean(lst):
            return _entropy_weighted_mean(np.array(lst))

        coarse_mu = {k: ew_mean(v) for k, v in coarse_data.items()}
        mid_mu    = {k: ew_mean(v) for k, v in mid_data.items()}
        fine_mu   = {k: ew_mean(v) for k, v in fine_data.items()}

        # --- Hierarchical shrinkage of means -----------------------------
        tau = self.tau

        # Shrink mid toward coarse
        mid_mu_shrunk = {}
        for k, mu in mid_mu.items():
            tc, db, co = k
            n = len(mid_data[k])
            parent_mu = coarse_mu.get((tc, co))
            if parent_mu is not None:
                lam = n / (n + tau)
                mid_mu_shrunk[k] = lam * mu + (1 - lam) * parent_mu
            else:
                mid_mu_shrunk[k] = mu

        # Shrink fine toward mid-shrunk
        fine_mu_shrunk = {}
        for k, mu in fine_mu.items():
            tc, db, co, den = k
            n = len(fine_data[k])
            parent_mu = mid_mu_shrunk.get((tc, db, co))
            if parent_mu is not None:
                lam = n / (n + tau)
                fine_mu_shrunk[k] = lam * mu + (1 - lam) * parent_mu
            else:
                fine_mu_shrunk[k] = mu

        # --- Fit α at fine level -----------------------------------------
        self.table = {}
        for k, mu_shrunk in fine_mu_shrunk.items():
            tc, db, co, den = k
            n = len(fine_data[k])
            kappa = self._fit_kappa(
                np.array(fine_data[k]), tc, co, n,
            )
            self.table[k] = kappa * mu_shrunk + self.epsilon

        # --- Fit fallback (tc, co) table ---------------------------------
        self.terrain_fallback = {}
        for (tc, co), mu in coarse_mu.items():
            n = len(coarse_data[(tc, co)])
            kappa = self._fit_kappa(
                np.array(coarse_data[(tc, co)]), tc, co, n,
            )
            self.terrain_fallback[(tc, co)] = kappa * mu + self.epsilon

        # Global fallback: uniform
        all_dists = np.concatenate(list(fine_data.values()), axis=0)
        global_mu = _entropy_weighted_mean(all_dists)
        self.global_fallback = global_mu * 1.0 + self.epsilon

        return self

    def _fit_kappa(
        self,
        distributions: np.ndarray,
        terrain_code:  int,
        coastal:       int,
        n:             int,
    ) -> float:
        """Estimate concentration κ with shrinkage toward terrain-group default.

        κ_final = λ · κ_MoM  +  (1−λ) · κ_group
        λ = n / (n + n_reg)

        The MoM estimate uses the median across classes, clipped to [kappa_min, kappa_max].
        """
        kappa_group = np.clip(
            _terrain_group_kappa(terrain_code, coastal),
            self.kappa_min, self.kappa_max,
        )

        mu  = distributions.mean(axis=0)
        var = distributions.var(axis=0)
        valid = var > 1e-8
        if valid.any():
            kappa_mom = float(np.median(
                mu[valid] * (1 - mu[valid]) / var[valid] - 1
            ))
            kappa_mom = np.clip(kappa_mom, self.kappa_min, self.kappa_max)
        else:
            kappa_mom = self.kappa_max

        lam = n / (n + self.n_reg)
        return float(lam * kappa_mom + (1 - lam) * kappa_group)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def lookup(
        self,
        terrain_code: int,
        dist_bin:     int,
        coastal:      int = 0,
        density_bin:  int = 0,
    ) -> np.ndarray:
        """Return Dirichlet α for one cell.

        Fallback hierarchy:
          1. (tc, db, co, den)  — exact fine bin
          2. (tc, co)           — terrain + coastal fallback
          3. global             — uniform last resort
        """
        key = (terrain_code, dist_bin, coastal, density_bin)
        if key in self.table:
            return self.table[key]
        tf = self.terrain_fallback.get((terrain_code, coastal))
        if tf is not None:
            return tf
        tf_alt = self.terrain_fallback.get((terrain_code, 1 - coastal))
        if tf_alt is not None:
            return tf_alt
        return self.global_fallback.copy()

    def build_prior(
        self,
        terrain:     np.ndarray,
        settlements: list[dict],
    ) -> np.ndarray:
        """Build (H, W, C) Dirichlet pseudo-count array for one seed.

        Args:
            terrain:     (H, W) terrain code grid.
            settlements: list of dicts with 'x', 'y' keys.

        Returns:
            (H, W, NUM_RAW_CODES) α array (8-state).
        """
        tc_map, db_map, _, co_map, den_map = compute_cell_features(
            terrain, settlements, self.dist_bins,
            self.density_bins, self.density_radius,
        )
        H, W = terrain.shape
        alpha = np.zeros((H, W, NUM_RAW_CODES), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                alpha[y, x] = self.lookup(
                    int(tc_map[y, x]),
                    int(db_map[y, x]),
                    int(co_map[y, x]),
                    int(den_map[y, x]),
                )
        return alpha

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the fitted lookup table to JSON."""
        data = {
            "kappa_min":      self.kappa_min,
            "kappa_max":      self.kappa_max,
            "epsilon":        self.epsilon,
            "tau":            self.tau,
            "n_reg":          self.n_reg,
            "dist_bins":      self.dist_bins.tolist(),
            "density_bins":   self.density_bins.tolist(),
            "density_radius": self.density_radius,
            "table": {
                ",".join(str(x) for x in k): v.tolist()
                for k, v in self.table.items()
            },
            "terrain_fallback": {
                f"{tc},{co}": v.tolist()
                for (tc, co), v in self.terrain_fallback.items()
            },
            "global_fallback": self.global_fallback.tolist(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DirichletLookup":
        """Load a fitted lookup table from JSON.

        Supports old 2-key and 3-key formats for backward compatibility.
        """
        data = json.loads(Path(path).read_text())
        obj = cls(
            kappa_min      = data["kappa_min"],
            kappa_max      = data["kappa_max"],
            epsilon        = data["epsilon"],
            tau            = data.get("tau",            10.0),
            n_reg          = data.get("n_reg",          50.0),
            dist_bins      = np.array(data["dist_bins"]),
            density_bins   = np.array(data.get("density_bins",   DEFAULT_DENSITY_BINS)),
            density_radius = int(data.get("density_radius", DEFAULT_DENSITY_RADIUS)),
        )

        obj.table = {}
        for key, alpha in data["table"].items():
            parts = [int(x) for x in key.split(",")]
            if len(parts) == 2:
                # Old 2-key: (tc, db) → treat as non-coastal, density=0
                obj.table[(parts[0], parts[1], 0, 0)] = np.array(alpha)
            elif len(parts) == 3:
                # Old 3-key: (tc, db, co) → density=0
                obj.table[(parts[0], parts[1], parts[2], 0)] = np.array(alpha)
            else:
                obj.table[tuple(parts)] = np.array(alpha)

        obj.terrain_fallback = {}
        for key, alpha in data["terrain_fallback"].items():
            parts = [int(x) for x in key.split(",")]
            if len(parts) == 1:
                obj.terrain_fallback[(parts[0], 0)] = np.array(alpha)
                obj.terrain_fallback[(parts[0], 1)] = np.array(alpha)
            else:
                obj.terrain_fallback[tuple(parts)] = np.array(alpha)

        obj.global_fallback = np.array(data["global_fallback"])
        return obj

    def summary(self) -> str:
        """Human-readable summary of the fitted lookup table."""
        n_coastal    = sum(1 for (_, _, co, _) in self.table if co == 1)
        n_noncoastal = sum(1 for (_, _, co, _) in self.table if co == 0)
        lines = [
            f"DirichletLookup: {len(self.table)} bins "
            f"({n_noncoastal} inland, {n_coastal} coastal), "
            f"{len(self.terrain_fallback)} terrain fallbacks",
            f"  κ range: [{self.kappa_min}, {self.kappa_max}]  "
            f"τ(mean)={self.tau}  n_reg(κ)={self.n_reg}  ε={self.epsilon}",
            f"  dist bins:    {self.dist_bins.tolist()}",
            f"  density bins: {self.density_bins.tolist()}  radius={self.density_radius}",
            "",
            "  (tc, db, coast, den) → α₀, top class:",
        ]
        for k, alpha in sorted(self.table.items())[:20]:
            tc, db, co, den = k
            total     = alpha.sum()
            top_class = int(alpha.argmax())
            top_prob  = alpha[top_class] / total
            lines.append(
                f"    ({tc:>2},{db},{'C' if co else 'I'},{den}) "
                f"→ α₀={total:6.1f}, top=class {top_class} ({top_prob:.1%})"
            )
        if len(self.table) > 20:
            lines.append(f"    … and {len(self.table) - 20} more bins")
        return "\n".join(lines)
