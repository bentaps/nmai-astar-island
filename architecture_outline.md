# Astar Island — Algorithm Architecture

## Competition Objective

For each cell `u` on a 40×40 map, predict a probability distribution `q_u ∈ Δ⁶` over 6 classes
(Empty, Settlement, Port, Ruin, Forest, Mountain).

**Scoring:**
```
weighted_KL  = Σ_u  H(p_u) · KL(p_u ‖ q_u)  /  Σ_u H(p_u)
score        = 100 · exp(−3 · weighted_KL)   ∈ [0, 100]   (higher = better)
```
where `p_u` is the ground truth, `H(p) = −Σ p_c ln p_c` is Shannon entropy.

Static cells (ocean, mountain) have `H(p_u) ≈ 0` and contribute **nothing** to the score.
Only dynamic cells (near settlements, coastal plains, etc.) matter.

**Implemented in:** [`astar/scoring.py`](astar/scoring.py) — `competition_score()`


---

## Pipeline Overview

```
Historical data  →  [1. Fit prior]        →  α_prior(u)
                                                 ↓
Active queries   →  [2. Adaptive queries] →  obs(u) + settlement_log
                                                 ↓
Regime inference →  [3. Regime]           →  expansion_rate, owner_entropy, …
                                                 ↓
Cross-seed pool  →  [4. Pool]             →  pooled(k)
                                                 ↓
Posterior update →  [5. Predict]          →  α_post(u) = α_prior(u) + obs(u) + pooled(k(u))
                                                 ↓
Calibration      →  [6. Calibrate]        →  temperature scaling + dynamic floor
                                                 ↓
Submit           →  [7. Submit]           →  q_u = α_scaled(u) / Σ α_scaled(u)
```


---

## Stage 1 — Dirichlet Prior (`DirichletLookup`)

**File:** [`astar/dirichlet.py`](astar/dirichlet.py) — `DirichletLookup.fit()` / `DirichletLookup.build_prior()`

Each cell is described by a **feature key** `k(u) = (terrain_code, dist_bin, coastal, density_bin)`.

**Features** ([`astar/features.py`](astar/features.py)):

| Feature | Description | Implementation |
|---------|-------------|----------------|
| `terrain_code` | Initial terrain type (0–5, 10, 11) | `terrain[y,x]` |
| `dist_bin` | Binned Manhattan distance to nearest settlement | `distance_to_nearest_settlement()` → `bin_distances()` |
| `coastal` | 1 if 4-connected adjacent to ocean (code 10) | `is_coastal()` |
| `density_bin` | Binned count of settlements within radius `r` | `settlement_density()` |

### 1a. Entropy-Weighted Mean

For each bin `k`, the target mean distribution is:
```
μ_k  =  Σ_{u: k(u)=k}  H(p_u) · p_u  /  Σ_{u: k(u)=k}  H(p_u)
```
Weights uncertain (dynamic) cells more than trivially static ones, aligning with the scoring rule.

### 1b. Hierarchical Shrinkage of Means

Fine bins may have few samples. The mean is shrunk toward a coarser parent bin:
```
λ     = n_k / (n_k + τ)
μ_shrunk(k) = λ · μ_k  +  (1−λ) · μ_parent(k)
```
Hierarchy: `(tc, db, co, den)` → `(tc, db, co)` → `(tc, co)`.

**Hyperparameter:**
- **`tau`** (default `20.0`) — regularisation strength. Higher = more shrinkage toward parent bin.

### 1c. Concentration Estimation (κ)

The Dirichlet concentration `κ_k` controls how strong the prior is:
- `α_k = κ_k · μ_shrunk_k + ε`
- `α₀_k = Σ_c α_k[c]` ≈ effective prior sample count

`κ` is estimated via Method of Moments from the empirical variance, then shrunk toward a terrain-group default:
```
κ_MoM  =  median_c[  μ_c(1−μ_c) / Var[p_c] − 1  ]
λ_κ    =  n / (n + n_reg)
κ_k    =  λ_κ · κ_MoM  +  (1−λ_κ) · κ_group(terrain, coastal)
```

**Hyperparameters:**
- **`n_reg`** (default `20.0`) — κ shrinkage strength. Lower = trust empirical κ more.
- **`kappa_min`** (default `2.0`) — floor on κ (prevents over-diffuse priors)
- **`kappa_max`** (default `200.0`) — ceiling on κ (prevents over-concentrated priors)
- **`epsilon`** (default `0.01`) — pseudo-count added to every class to prevent zero probabilities

**Terrain-group κ defaults** (hardcoded in `astar/dirichlet.py`):

| Terrain | κ_group | Meaning |
|---------|---------|---------|
| Ocean (10), Mountain (5) | 200 | Never changes — very strong prior |
| Forest (4) | 15 | Mostly static |
| Plains (11), Empty (0) | 5 | Dynamic frontier |
| Settlement (1), Ruin (3) | 4 | Dynamic |
| Port (2) | 3 | Very dynamic (coastal) |

Coastal non-static cells get their κ multiplied by `_COASTAL_KAPPA_FACTOR = 0.6`, making coastal cells weaker (more responsive to observations).

### 1d. Dist and Density Bins

**Hyperparameters:**
- **`dist_bins`** (default `[0,1,2,3,4,6,8,12,20,9999]`) — bin edges for distance to nearest settlement
- **`density_bins`** (default `[0,1,2,4,9999]`) — bin edges for settlement density
- **`density_radius`** (default `6`) — Manhattan radius for counting nearby settlements


---

## Stage 2 — Adaptive Multi-Pass Queries

**File:** [`astar/inference.py`](astar/inference.py) — `run_adaptive_queries()`

Instead of querying all seeds uniformly, the adaptive strategy front-loads observations on the
most informative seeds to maximise cross-seed pooling leverage.

### 2a. Phase 1 — Diagnostic (40% budget)

Pick the 2 seeds with the most dynamic cells. Query them thoroughly with greedy set-cover viewports.
Cross-seed pooling then propagates this information to all other seeds.

### 2b. Phase 2 — Spread (40% budget)

Query remaining seeds using `dynamic_coverage_viewports()`. Viewport order is guided by
**interim entropy**: build pooled predictions after Phase 1, then rank dynamic viewports
by total entropy of the interim posterior.

### 2c. Phase 3 — Refine (remaining budget)

Spend all remaining queries on windows ranked by **posterior entropy** across all seeds
(`rank_windows_by_entropy()`). Top 3 candidates per seed are pooled globally and the
highest-entropy windows execute first.

### Fallback: Uniform Coverage

The standard (non-adaptive) strategy divides the budget equally: `vps_per_seed = (budget − resample_budget) // num_seeds`,
followed by entropy-ranked resampling. Toggle via `--adaptive` flag in scripts.

**Budget split (adaptive):** 40% diagnostic | 40% spread | remaining refine
**Budget split (uniform):** 90% coverage | 10% resampling

**Hyperparameters:**
- **`max_settlement_dist`** (default `12`) — cells farther than this from any settlement are treated as static
- **`vp`** (default `15`) — viewport size (matches competition API maximum)
- **`stride`** (default `5`) — grid spacing for candidate viewport positions


---

## Stage 3 — Regime Inference

**File:** [`astar/regime.py`](astar/regime.py) — `estimate_regime()` / `regime_summary()`

The `/simulate` endpoint returns a `settlements` field in each response. These payloads contain
rich per-settlement metadata: `population`, `food`, `wealth`, `defense`, `has_port`, `alive`, `owner_id`.

By comparing the observed settlement state against the initial settlements, we infer the round's
**expansion regime**:

```
expansion_rate  =  (observed_count − initial_count) / initial_count
```

Other features extracted: `port_fraction`, `mean_population`, `mean_defense`, `owner_entropy`
(Shannon entropy of owner_id distribution), `alive_fraction`.

**Regime categories:** heavy expansion (>+50%) | moderate expansion | stable | decline | collapse

**Usage:** Currently logged for diagnostics. Future work: use regime features to adjust
`pool_weight` and `temperature` per round, or select regime-specific calibration.

**Offline simulation:** `SimulatedSession` accepts a `settlement_records` dict loaded from
`data/rounds/{id}/settlement_records.json` (available for rounds 6 and 8) to enable offline
regime testing.


---

## Stage 4 — Cross-Seed Feature Pooling

**File:** [`astar/inference.py`](astar/inference.py) — `pool_observations_by_feature()`

After coverage queries, each cell has at most 1 observation. With prior `α₀ ≈ 4`, one observation
has weight `1/(4+1) = 20%`.

**Key insight:** the prior already assumes all cells with the same feature key `k` share the same
distribution. We can therefore pool observations across cells and seeds:
```
pooled(k)  =  Σ_{seeds s}  Σ_{u: k(u)=k}  obs_s(u)
```
A bin with 5 seeds × 20 cells in the same bin gives ~100 pooled observations → weight `100/(4+100) ≈ 96%`.

**No new hyperparameters** beyond `pool_weight` below.


---

## Stage 5 — Posterior Update

**File:** [`astar/inference.py`](astar/inference.py) — `build_predictions_pooled()`

The Dirichlet-multinomial conjugate update:
```
α_post(u)  =  α_prior(u)  +  obs(u)  +  pool_weight · (pooled(k(u)) − obs(u))
```
The `−obs(u)` term subtracts this cell's own contribution to avoid double-counting.

**Hyperparameter:**
- **`pool_weight`** (default `1.5`) — scaling factor for pooled evidence. `1.0` = full pooling.
  Values >1 upweight pooled cross-seed evidence relative to the prior.


---

## Stage 6 — Calibration (Temperature + Dynamic Floor)

**File:** [`astar/inference.py`](astar/inference.py) — `dynamic_floor()` inside `build_predictions_pooled()`

### 6a. Temperature Scaling

Applied to the posterior before normalisation:
```
α_scaled(u)  =  α_post(u) ^ (1/T)
q(u)         =  α_scaled(u) / Σ_c α_scaled(u)[c]
```
- `T < 1` → sharper predictions (concentrate mass on the leading class)
- `T > 1` → softer predictions (more uniform)
- `T = 1` → no change (current default)

**Hyperparameter:** **`temperature`** (default `1.0`)

### 6b. Dynamic Floor

Replaces the old fixed probability floor with a per-cell adaptive floor:
```
floor(u)  =  floor_base  +  floor_obs_scale / (1 + n_obs(u))
```
where `n_obs(u) = Σ_c (α_post(u) − α_prior(u))[c]` is the total pooled evidence for cell `u`.

- Well-observed cells → small floor → sharp, trusted predictions
- Unobserved cells → larger floor → safe smoothing against zero-probability errors

**Hyperparameters:**
- **`floor_base`** (default `0.002`) — minimum floor even for well-observed cells
- **`floor_obs_scale`** (default `0.02`) — additional floor for unobserved cells


---

## Complete Hyperparameter Table

| Parameter | Default | Stage | Where set |
|-----------|---------|-------|-----------|
| `tau` | `20.0` | Prior mean shrinkage | `DirichletLookup(tau=)` |
| `n_reg` | `20.0` | κ shrinkage strength | `DirichletLookup(n_reg=)` |
| `kappa_min` | `2.0` | κ floor | `DirichletLookup(kappa_min=)` |
| `kappa_max` | `200.0` | κ ceiling | `DirichletLookup(kappa_max=)` |
| `epsilon` | `0.01` | α pseudo-count floor | `DirichletLookup(epsilon=)` |
| `dist_bins` | `[0,1,2,3,4,6,8,12,20,9999]` | Feature binning | `DirichletLookup(dist_bins=)` |
| `density_bins` | `[0,1,2,4,9999]` | Feature binning | `DirichletLookup(density_bins=)` |
| `density_radius` | `6` | Feature binning | `DirichletLookup(density_radius=)` |
| `max_settlement_dist` | `12` | Dynamic cell mask | `run_adaptive_queries(max_settlement_dist=)` |
| `pool_weight` | `1.5` | Posterior pooling | `build_predictions_pooled(pool_weight=)` |
| `temperature` | `1.0` | Posterior sharpening | `build_predictions_pooled(temperature=)` |
| `floor_base` | `0.002` | Dynamic floor minimum | `build_predictions_pooled(floor_base=)` |
| `floor_obs_scale` | `0.02` | Dynamic floor decay | `build_predictions_pooled(floor_obs_scale=)` |

Defaults for `tau`, `n_reg`, and `pool_weight` were tuned via grid search on rounds 6–8.


---

## Data Flow Summary

```
data/X_initial.npy          (N, 40, 40)  terrain codes
data/Y_ground_truth.npy     (N, 40, 40, 6)  ground truth distributions
data/training_meta.json     [{round_num, round_id, seed_idx, score}]

         ↓  DirichletLookup.fit()  [astar/dirichlet.py]
         ↓  features: compute_cell_features()  [astar/features.py]

table: {(tc,db,co,den) → α (6,)}   ~125 bins

         ↓  DirichletLookup.build_prior()

α_prior  (5 seeds, 40, 40, 6)

         ↓  run_adaptive_queries()  [astar/inference.py]
         ↓    Phase 1: diagnostic (2 seeds, 40% budget)
         ↓    Phase 2: spread (3 seeds, guided by interim entropy)
         ↓    Phase 3: refine (entropy-ranked resampling)

obs      {seed → (40, 40, 6)}   observation counts
settlement_log  [list[dict]]    settlement payloads per query

         ↓  estimate_regime()  [astar/regime.py]

regime   {expansion_rate, port_fraction, mean_population, owner_entropy, …}

         ↓  pool_observations_by_feature()

pooled   {(tc,db,co,den) → (6,)}  summed across all observed cells+seeds

         ↓  build_predictions_pooled()
         ↓    posterior: α_post = α_prior + obs + pool_weight * pooled_other
         ↓    temperature scaling: α_scaled = α_post ^ (1/T)
         ↓    dynamic floor: floor(u) = floor_base + floor_obs_scale / (1 + n_obs(u))

q_final  (5 seeds, 40, 40, 6)   floored & normalised probability maps

         ↓  competition_score()  [astar/scoring.py]

score    float ∈ [0, 100]
```


---

## Key Files

| File | Role |
|------|------|
| [`astar/dirichlet.py`](astar/dirichlet.py) | Dirichlet lookup table: fit, shrinkage, prior building |
| [`astar/features.py`](astar/features.py) | Feature extraction: terrain codes, distance/density bins |
| [`astar/inference.py`](astar/inference.py) | Query planning, pooling, prediction building, adaptive strategy |
| [`astar/regime.py`](astar/regime.py) | Regime inference from settlement payloads |
| [`astar/scoring.py`](astar/scoring.py) | Competition score, KL divergence, floor-and-normalize |
| [`astar/session.py`](astar/session.py) | SimulatedSession (offline) with optional settlement records |
| [`astar/api.py`](astar/api.py) | LiveSession and API helpers |
| [`simulate_round.py`](simulate_round.py) | Offline replay: fit → query → score → plot |
| [`submit_solution.py`](submit_solution.py) | Live submission: prior → adaptive queries → final submit |
| [`gridsearch.py`](gridsearch.py) | Hyperparameter grid search (supports `--adaptive`) |
