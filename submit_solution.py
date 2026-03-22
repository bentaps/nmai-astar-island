#!/usr/bin/env python3
"""
Astar Island — NM i AI 2026 Solution

Bayesian active inference over a black-box Norse civilization simulator.
All logic lives in the astar/ package. This file is the CLI entry point.

WARNING: Running this script makes REAL API calls and submits to the active
round. Only run this when you intend to make an official submission.
"""

import os
from pathlib import Path

from astar.api import (
    RateLimitedSession,       # used by historical_data.ipynb
    LiveSession,
    get_rounds,
    get_round_details,
    get_budget,
    get_my_rounds,
    get_analysis,             # used by historical_data.ipynb
)
from astar.data    import parse_initial_states, save_query_log, save_round_data
from astar.dirichlet  import DirichletLookup
from astar.evaluation import load_dataset
from astar.features   import collapse_8state_to_6class
from astar.inference  import (
    build_predictions_pooled,
    dynamic_coverage_viewports,
    run_adaptive_queries,
    run_coverage_queries,
    run_resampling_queries,
)
from astar.regime     import estimate_regime, regime_adjustments, regime_summary
from astar.scoring    import apply_floor_and_normalize

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Calibration hyperparameters (tuned via gridsearch.py)
# ---------------------------------------------------------------------------
USE_ADAPTIVE    = False  # Set True to use adaptive multi-pass (experimental)
NEURAL_BLEND    = 0.40   # 0.0 = pure v3, 1.0 = pure neural, 0.4 = recommended blend
POOL_WEIGHT     = 1.581
TEMPERATURE     = 1.041
FLOOR_BASE      = 0.0002143
FLOOR_OBS_SCALE = 0.008436


def main() -> None:
    token = os.environ.get("AINM_TOKEN", "")
    if not token:
        print("ERROR: Set AINM_TOKEN environment variable")
        print("  export AINM_TOKEN='your-jwt-token-here'")
        return

    http = RateLimitedSession(token)

    print("=" * 60)
    print("ASTAR ISLAND — Bayesian Active Inference Solution")
    print("=" * 60)

    # ── Find active round ──────────────────────────────────────────────
    rounds = get_rounds(http)
    active_rounds = [r for r in rounds if r["status"] == "active"]
    if not active_rounds:
        print("No active rounds found.")
        for r in rounds:
            print(f"  Round {r['round_number']}: {r['status']} (id={r['id']})")
        completed = [r for r in rounds if r["status"] in ("completed", "closed")]
        if completed:
            print("\nFetching ground truth for completed rounds...")
            from astar.data import fetch_and_save_ground_truth
            for r in completed:
                print(f"\n  Round {r['round_number']} ({r['id']}):")
                fetch_and_save_ground_truth(http, r["id"], r.get("seeds_count", 5), get_analysis)
        return

    round_info = active_rounds[0]
    round_id   = round_info["id"]
    print(f"\nActive round: #{round_info['round_number']} (id={round_id})")
    print(f"  Map: {round_info['map_width']}x{round_info['map_height']}")
    print(f"  Closes at: {round_info.get('closes_at', 'unknown')}")

    # ── Round details ──────────────────────────────────────────────────
    print("\nFetching round details...")
    round_data = get_round_details(http, round_id)
    H          = round_data["map_height"]
    W          = round_data["map_width"]
    num_seeds  = round_data.get("seeds_count", len(round_data["initial_states"]))
    print(f"  Seeds: {num_seeds}, Grid: {H}x{W}")

    terrains, settlements_list = parse_initial_states(round_data)
    for i, (_, s) in enumerate(zip(terrains, settlements_list)):
        n_ports = sum(1 for ss in s if ss.get("has_port"))
        print(f"  Seed {i}: {len(s)} settlements, {n_ports} ports")

    live = LiveSession(token, round_id)

    # ── Fit prior on all historical rounds ─────────────────────────────
    print("\n--- Fitting prior on all historical rounds ---")
    dataset = load_dataset(DATA_DIR)
    lookup  = DirichletLookup()
    all_idx = list(range(len(dataset["X"])))
    lookup.fit(
        dataset["X"][all_idx],
        dataset["Y"][all_idx],
        [dataset["settlements"][i] for i in all_idx],
    )
    print(f"  Fitted on {len(all_idx)} samples from rounds {dataset['round_numbers']}")
    alphas = [lookup.build_prior(terrains[i], settlements_list[i]) for i in range(num_seeds)]

    # ── Phase 1: Submit baseline (prior only) ──────────────────────────
    print(f"\n--- PHASE 1: Submit baseline (data-driven priors) ---")
    baseline = [
        apply_floor_and_normalize(
            collapse_8state_to_6class(a / a.sum(axis=-1, keepdims=True))
        )
        for a in alphas
    ]
    for seed_idx, pred in enumerate(baseline):
        result = live.submit(seed_idx, pred)
        print(f"  Seed {seed_idx}: submitted ({result.get('status', 'ok')})")

    # ── Phase 2–4: Queries ──────────────────────────────────────────────
    budget_data  = get_budget(http, round_id)
    total_budget = budget_data["queries_max"]
    strategy = "adaptive" if USE_ADAPTIVE else "uniform"
    print(f"\n--- PHASES 2-4: {strategy} queries (budget={total_budget}) ---")

    if USE_ADAPTIVE:
        observations, extra_obs, settlement_log = run_adaptive_queries(
            live, num_seeds, terrains, settlements_list, alphas,
            budget=total_budget,
            pool_weight=POOL_WEIGHT,
            temperature=TEMPERATURE,
            floor_base=FLOOR_BASE,
            floor_obs_scale=FLOOR_OBS_SCALE,
            map_size=(H, W),
        )
    else:
        resample_budget = max(5, total_budget // 10)
        vps_per_seed = (total_budget - resample_budget) // num_seeds

        viewports_per_seed = [
            dynamic_coverage_viewports(
                terrains[s], settlements_list[s], max_viewports=vps_per_seed,
                max_settlement_dist=8,
            )
            for s in range(num_seeds)
        ]
        for s, vps in enumerate(viewports_per_seed):
            print(f"  Seed {s}: {len(vps)} targeted viewports")

        observations, _, settlement_log = run_coverage_queries(
            live, num_seeds, viewports_per_seed, (H, W),
        )

        remaining   = live.budget_remaining
        alphas_post = [alphas[s] + observations[s] for s in range(num_seeds)]
        print(f"  Resampling with {remaining} remaining queries...")
        extra_obs, _ = run_resampling_queries(
            live, num_seeds, terrains, settlements_list, remaining,
            alphas_post=alphas_post, map_size=(H, W),
            settlement_log=settlement_log,
        )

    # Regime inference from settlement payloads
    regime = estimate_regime(settlement_log, settlements_list)
    print(f"\n  Regime: {regime_summary(regime)}")

    adj = regime_adjustments(regime)
    final_pool_weight = POOL_WEIGHT * adj["pool_weight_mult"]
    final_floor_base  = FLOOR_BASE * adj["floor_base_mult"]
    final_temperature = TEMPERATURE + adj["temperature_adj"]
    if adj != {"pool_weight_mult": 1.0, "floor_base_mult": 1.0, "temperature_adj": 0.0}:
        print(f"  Regime adjustments: pool_weight={final_pool_weight:.2f}, "
              f"floor_base={final_floor_base:.4f}, temperature={final_temperature:.2f}")

    # ── Phase 5: Final predictions + submission ────────────────────────
    print("\n--- PHASE 5: Final submission ---")
    final = build_predictions_pooled(
        alphas, observations, terrains, settlements_list,
        extra_obs=extra_obs,
        pool_weight=final_pool_weight,
        temperature=final_temperature,
        floor_base=final_floor_base,
        floor_obs_scale=FLOOR_OBS_SCALE,
    )

    # Optional blended neural correction on top of v3 baseline
    if NEURAL_BLEND > 0:
        try:
            from astar.neural.predict import load_predictor
            neural = load_predictor()
        except Exception:
            neural = None
        if neural is not None:
            print(f"  Applying neural blend ({NEURAL_BLEND:.0%} neural + {1-NEURAL_BLEND:.0%} v3)...")
            neural_preds = neural.correct_predictions(
                final, terrains, settlements_list, observations, extra_obs,
            )
            final = [
                (1.0 - NEURAL_BLEND) * b + NEURAL_BLEND * n
                for b, n in zip(final, neural_preds)
            ]
            print("  Blended predictions ready.")
        else:
            print("  No neural model found, using v3 baseline.")

    for seed_idx, pred in enumerate(final):
        result = live.submit(seed_idx, pred)
        print(f"  Seed {seed_idx}: final submission ({result.get('status', 'ok')})")

    # ── Results ────────────────────────────────────────────────────────
    print("\n--- Results ---")
    try:
        my_rounds = get_my_rounds(http)
        for r in my_rounds:
            if isinstance(r, dict) and r.get("id") == round_id:
                print(f"  Round score: {r.get('round_score', 'pending')}")
                print(f"  Seed scores: {r.get('seed_scores', 'pending')}")
                print(f"  Rank: {r.get('rank', '?')}/{r.get('total_teams', '?')}")
                break
    except Exception as e:
        print(f"  Could not fetch scores yet: {e}")

    # ── Save data ──────────────────────────────────────────────────────
    print("\n--- Saving data ---")
    save_round_data(round_id, round_data, observations, final, extra_obs)
    save_query_log(round_id, settlement_log)

    budget_data = get_budget(http, round_id)
    print(f"\nFinal budget: {budget_data['queries_used']}/{budget_data['queries_max']} queries used")
    print("Done.")


if __name__ == "__main__":
    main()
