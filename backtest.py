#!/usr/bin/env python3
"""
Rolling backtest — backtest.py

Runs the full pipeline (prior fit + queries + scoring) across a rolling window
of rounds, always training only on rounds strictly before the test round.
Optionally compares against a saved baseline and exits nonzero on regression.

Usage:
    python backtest.py                             # default rounds 6-15
    python backtest.py --rounds 10 11 12 13 14 15  # specific rounds
    python backtest.py --n-seeds 3                 # average over 3 RNG seeds
    python backtest.py --save-baseline              # save current as baseline
    python backtest.py --threshold 1.0              # fail if any round drops >1pt
"""

# ── USER CONFIG ─────────────────────────────────────────────────────────────
TEST_ROUNDS     = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
BUDGET          = 50
RNG_SEED_BASE   = 42
N_RNG_SEEDS     = 1       # average over multiple MC seeds for stability
BASELINE_PATH   = "data/backtest_baseline.json"
REGRESSION_THRESHOLD = 0.5  # fail if any round drops by more than this
# ────────────────────────────────────────────────────────────────────────────

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from astar.data       import parse_initial_states
from astar.dirichlet  import DirichletLookup
from astar.evaluation import load_dataset
from astar.inference  import (
    build_predictions_pooled,
    dynamic_coverage_viewports,
    run_coverage_queries,
    run_resampling_queries,
)
from astar.regime     import estimate_regime, regime_adjustments, regime_summary
from astar.scoring    import competition_score
from astar.session    import SimulatedSession

DATA_DIR = Path("data")

# Calibration defaults (must match simulate_round.py / submit_solution.py)
POOL_WEIGHT     = 1.581
TEMPERATURE     = 1.041
FLOOR_BASE      = 0.0002143
FLOOR_OBS_SCALE = 0.008436


def evaluate_round(
    round_num: int,
    dataset: dict,
    budget: int,
    rng_seed: int,
    pool_weight: float = POOL_WEIGHT,
    temperature: float = TEMPERATURE,
    floor_base: float = FLOOR_BASE,
    floor_obs_scale: float = FLOOR_OBS_SCALE,
) -> dict | None:
    """Run the full pipeline for one round. Returns None if round not available."""
    meta = dataset["meta"]
    X    = dataset["X"]
    Y    = dataset["Y"]       # 8-state for training/simulation
    Y6   = dataset["Y6"]      # 6-class for scoring
    setts = dataset["settlements"]

    train_idx = [i for i, m in enumerate(meta) if m["round_num"] < round_num]
    if not train_idx:
        return None

    lookup = DirichletLookup()
    lookup.fit(X[train_idx], Y[train_idx], [setts[i] for i in train_idx])

    test_idx = sorted(
        [i for i, m in enumerate(meta) if m["round_num"] == round_num],
        key=lambda i: meta[i]["seed_idx"],
    )
    if not test_idx:
        return None

    round_id = meta[test_idx[0]]["round_id"]
    rd_path  = DATA_DIR / "rounds" / round_id / "round_data.json"
    if not rd_path.exists():
        return None

    round_data           = json.loads(rd_path.read_text())
    terrains, setts_list = parse_initial_states(round_data)
    H, W                 = terrains[0].shape
    ground_truth         = [Y[i] for i in test_idx]     # 8-state for SimulatedSession
    ground_truth_6       = [Y6[i] for i in test_idx]    # 6-class for scoring
    num_seeds            = len(test_idx)

    alphas = [lookup.build_prior(terrains[s], setts_list[s]) for s in range(num_seeds)]

    # Load settlement records if available
    sett_records_path = DATA_DIR / "rounds" / round_id / "settlement_records.json"
    sett_records = (json.loads(sett_records_path.read_text())
                    if sett_records_path.exists() else None)

    sim = SimulatedSession(ground_truth, budget=budget, rng_seed=rng_seed,
                           settlement_records=sett_records)

    # Uniform coverage + resampling
    resample_budget = max(5, budget // 10)
    vps_per_seed = (budget - resample_budget) // num_seeds

    viewports_per_seed = [
        dynamic_coverage_viewports(
            terrains[s], setts_list[s], max_viewports=vps_per_seed,
            max_settlement_dist=8,
        )
        for s in range(num_seeds)
    ]
    observations, _, settlement_log = run_coverage_queries(
        sim, num_seeds, viewports_per_seed, (H, W), verbose=False,
    )
    remaining   = sim.budget_remaining
    alphas_post = [alphas[s] + observations[s] for s in range(num_seeds)]
    extra_obs, _ = run_resampling_queries(
        sim, num_seeds, terrains, setts_list, remaining,
        alphas_post=alphas_post, map_size=(H, W), verbose=False,
        settlement_log=settlement_log,
    )

    # Regime-conditioned calibration
    regime = estimate_regime(settlement_log, setts_list)
    adj = regime_adjustments(regime)
    final_pw = pool_weight * adj["pool_weight_mult"]
    final_fb = floor_base * adj["floor_base_mult"]
    final_t  = temperature + adj["temperature_adj"]

    preds = build_predictions_pooled(
        alphas, observations, terrains, setts_list,
        extra_obs=extra_obs,
        pool_weight=final_pw,
        temperature=final_t,
        floor_base=final_fb,
        floor_obs_scale=floor_obs_scale,
    )

    scores = [competition_score(ground_truth_6[s], preds[s]) for s in range(num_seeds)]
    return {
        "round": round_num,
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "regime": regime_summary(regime),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds", type=int, nargs="+", default=TEST_ROUNDS)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--n-seeds", type=int, default=N_RNG_SEEDS,
                        help="Number of RNG seeds to average over")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current results as the baseline")
    parser.add_argument("--threshold", type=float, default=REGRESSION_THRESHOLD,
                        help="Fail if any round regresses by more than this")
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = load_dataset(DATA_DIR)
    print(f"  {len(dataset['X'])} samples, rounds {dataset['round_numbers']}\n")

    print(f"Rolling backtest: rounds {args.rounds}, "
          f"budget={args.budget}, n_rng_seeds={args.n_seeds}")
    print(f"{'='*70}")

    results = {}
    t0 = time.time()

    for round_num in args.rounds:
        seed_results = []
        for rng_offset in range(args.n_seeds):
            rng_seed = RNG_SEED_BASE + rng_offset
            r = evaluate_round(round_num, dataset, args.budget, rng_seed)
            if r is not None:
                seed_results.append(r)

        if not seed_results:
            print(f"  Round {round_num:>3}: SKIPPED (no data)")
            continue

        means = [r["mean"] for r in seed_results]
        overall_mean = float(np.mean(means))
        overall_std  = float(np.std(means)) if len(means) > 1 else 0.0
        per_seed     = seed_results[0]["scores"]
        regime_str   = seed_results[0]["regime"]

        results[round_num] = {
            "mean": overall_mean,
            "std": overall_std,
            "per_seed": per_seed,
            "regime": regime_str,
        }

        seed_str = "  ".join(f"{s:.1f}" for s in per_seed)
        std_str = f" ± {overall_std:.2f}" if args.n_seeds > 1 else ""
        print(f"  Round {round_num:>3}: {overall_mean:5.1f}{std_str}  "
              f"[{seed_str}]  {regime_str}")

    elapsed = time.time() - t0
    all_means = [v["mean"] for v in results.values()]
    grand_mean = float(np.mean(all_means)) if all_means else 0.0

    print(f"{'='*70}")
    print(f"  Grand mean: {grand_mean:.1f}/100  ({elapsed:.1f}s)")

    # Save baseline if requested
    if args.save_baseline:
        Path(BASELINE_PATH).parent.mkdir(exist_ok=True)
        with open(BASELINE_PATH, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Baseline saved to {BASELINE_PATH}")

    # Compare against baseline if it exists
    baseline_path = Path(BASELINE_PATH)
    if baseline_path.exists() and not args.save_baseline:
        baseline = json.loads(baseline_path.read_text())
        print(f"\n  Comparison vs baseline ({BASELINE_PATH}):")
        regressions = []
        for rn in args.rounds:
            rn_str = str(rn)
            if rn_str not in baseline or rn not in results:
                continue
            base_mean = baseline[rn_str]["mean"]
            curr_mean = results[rn]["mean"]
            delta = curr_mean - base_mean
            marker = "  REGRESSION" if delta < -args.threshold else ""
            print(f"    Round {rn:>3}: {curr_mean:5.1f} vs {base_mean:5.1f}  "
                  f"({delta:+.1f}){marker}")
            if delta < -args.threshold:
                regressions.append(rn)

        if regressions:
            print(f"\n  FAIL: Regressions detected on rounds {regressions} "
                  f"(threshold={args.threshold})")
            sys.exit(1)
        else:
            print(f"\n  PASS: No regressions beyond threshold ({args.threshold})")

    print()


if __name__ == "__main__":
    main()
