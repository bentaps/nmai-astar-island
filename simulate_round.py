#!/usr/bin/env python3
"""
Simulated round replay — simulate_round.py

Fits a DirichletLookup prior on all rounds BEFORE the chosen round,
then runs the full query pipeline (coverage + resampling) via a
SimulatedSession and scores against the held-out ground truth.

If the requested round is not in the local dataset, the script will
automatically fetch it from the API (requires AINM_TOKEN env var),
rebuild the training arrays, and proceed.

Usage:
    python simulate_round.py          # uses ROUND_NUM below
    python simulate_round.py --round 4
"""

# ── USER CONFIG ────────────────────────────────────────────────────────────────
ROUND_NUM       = 22    # ← change to the round you want to replay
BUDGET          = 50    # query budget (same as live competition)
RNG_SEED        = 42    # reproducibility
SHOW_FIG        = False # set True to display figures interactively

# Calibration hyperparameters (tuned via gridsearch.py on rounds 10-14)
POOL_WEIGHT     = 1.581   # cross-seed pooling strength
TEMPERATURE     = 1.041   # posterior sharpening (<1) or softening (>1)
FLOOR_BASE      = 0.0002143 # minimum floor per cell
FLOOR_OBS_SCALE = 0.008436  # floor shrinkage rate with observations
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from astar.data       import parse_initial_states
from astar.dirichlet  import DirichletLookup
from astar.evaluation import load_dataset, rebuild_training_arrays
from astar.inference  import (
    build_predictions_pooled,
    dynamic_coverage_viewports,
    run_adaptive_queries,
    run_coverage_queries,
    run_resampling_queries,
)
from astar.features  import collapse_8state_to_6class
from astar.regime    import estimate_regime, regime_adjustments, regime_summary
from astar.scoring   import apply_floor_and_normalize, competition_score
from astar.session   import SimulatedSession
from astar.visualise import CLASS_NAMES, CLASS_COLORS

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# API sync helpers
# ---------------------------------------------------------------------------

def _fetch_round_from_api(round_num: int) -> bool:
    """Fetch round data + analysis from the live API and save to data/rounds/.

    Returns True if ground truth was successfully fetched for at least one seed.
    Requires AINM_TOKEN environment variable.
    """
    token = os.environ.get("AINM_TOKEN", "")
    if not token:
        print("  ✗ AINM_TOKEN not set — cannot fetch data automatically.")
        print("    Set it with:  export AINM_TOKEN='your-jwt-token-here'")
        return False

    from astar.api import RateLimitedSession, get_rounds, get_round_details, get_analysis

    print(f"  Connecting to API...")
    http   = RateLimitedSession(token)
    rounds = get_rounds(http)
    print(f"  Found {len(rounds)} rounds on server: "
          f"{[r['round_number'] for r in rounds]}")

    target = next((r for r in rounds if r["round_number"] == round_num), None)
    if target is None:
        print(f"  ✗ Round {round_num} does not exist on the server.")
        print(f"    Available round numbers: {sorted(r['round_number'] for r in rounds)}")
        return False

    round_id  = target["id"]
    status    = target.get("status", "unknown")
    print(f"  Round {round_num}: id={round_id}  status={status}")

    # --- round_data.json -------------------------------------------------
    round_dir = DATA_DIR / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    rd_path = round_dir / "round_data.json"
    if rd_path.exists():
        print(f"  round_data.json already cached — skipping fetch.")
        rd = json.loads(rd_path.read_text())
    else:
        print(f"  Fetching round details from API...")
        rd = get_round_details(http, round_id)
        rd_path.write_text(json.dumps(rd, indent=2))
        num_states = len(rd.get("initial_states", []))
        H = rd.get("map_height", "?")
        W = rd.get("map_width",  "?")
        print(f"  ✓ Saved round_data.json  ({num_states} seeds, {H}×{W} map)")

    # --- analysis_seed{i}.json  (ground truth) ---------------------------
    num_seeds = rd.get("seeds_count", len(rd.get("initial_states", [])))
    fetched   = 0

    for seed_idx in range(num_seeds):
        path = round_dir / f"analysis_seed{seed_idx}.json"
        if path.exists():
            analysis = json.loads(path.read_text())
            gt_ok = analysis.get("ground_truth") is not None
            print(f"  Seed {seed_idx}: analysis already cached  "
                  f"(ground_truth={'present' if gt_ok else 'MISSING'})")
            if gt_ok:
                fetched += 1
            continue

        print(f"  Seed {seed_idx}: fetching analysis...", end=" ", flush=True)
        try:
            analysis = get_analysis(http, round_id, seed_idx)
            gt_ok    = analysis.get("ground_truth") is not None
            score    = analysis.get("score")
            path.write_text(json.dumps(analysis, indent=2))
            if gt_ok:
                fetched += 1
                print(f"✓  score={score:.2f}" if score is not None else "✓  (score pending)")
            else:
                print(f"⚠  saved but ground_truth is null "
                      f"(round may still be active or we never submitted)")
        except Exception as e:
            print(f"✗  {e}")

    if fetched == 0:
        print(f"\n  ✗ No ground truth available for round {round_num}.")
        if status == "active":
            print("    The round is still active — ground truth will only be "
                  "available after it closes.")
        else:
            print("    We may not have submitted to this round, so the analysis "
                  "endpoint returns null.")
        return False

    print(f"  ✓ Ground truth available for {fetched}/{num_seeds} seeds.")
    return True


def _ensure_round_in_dataset(round_num: int) -> dict | None:
    """Load dataset, auto-fetching from API if round_num is missing.

    Returns the loaded dataset dict, or None if the round cannot be obtained.
    """
    dataset    = load_dataset(DATA_DIR)
    all_rounds = sorted(dataset["round_numbers"])

    if round_num in all_rounds:
        return dataset

    print(f"\nRound {round_num} not in local dataset (have: {all_rounds}).")
    print(f"Attempting to fetch from API...\n")

    success = _fetch_round_from_api(round_num)
    if not success:
        return None

    # Rebuild numpy training arrays from raw JSON
    print(f"\nRebuilding training arrays from data/rounds/...")
    n = rebuild_training_arrays(DATA_DIR)
    print(f"✓ Rebuilt: {n} samples across all rounds.")

    # Reload
    dataset    = load_dataset(DATA_DIR)
    all_rounds = sorted(dataset["round_numbers"])
    print(f"Dataset now contains rounds: {all_rounds}\n")

    if round_num not in all_rounds:
        print(f"✗ Round {round_num} still missing after rebuild — "
              f"cannot proceed.")
        return None

    return dataset


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    round_num: int,
    budget: int,
    rng_seed: int,
    show_fig: bool = SHOW_FIG,
    pool_weight: float = POOL_WEIGHT,
    temperature: float = TEMPERATURE,
    floor_base: float = FLOOR_BASE,
    floor_obs_scale: float = FLOOR_OBS_SCALE,
    adaptive: bool = False,
) -> None:

    # ------------------------------------------------------------------
    # 1. Load dataset (auto-fetching if needed)
    # ------------------------------------------------------------------
    dataset = _ensure_round_in_dataset(round_num)
    if dataset is None:
        return

    meta        = dataset["meta"]
    X           = dataset["X"]
    Y           = dataset["Y"]       # 8-state for training/simulation
    Y6          = dataset["Y6"]      # 6-class for scoring
    settlements = dataset["settlements"]
    all_rounds  = sorted(dataset["round_numbers"])

    print(f"Dataset loaded: {len(meta)} samples across rounds {all_rounds}")

    # ------------------------------------------------------------------
    # 2. Fit prior on rounds STRICTLY before round_num
    # ------------------------------------------------------------------
    train_idx = [i for i, m in enumerate(meta) if m["round_num"] < round_num]
    if not train_idx:
        print(f"\n✗ No training data before round {round_num}. Cannot fit prior.")
        print(f"  (need at least one completed round with ground truth)")
        return

    train_rounds = sorted({meta[i]["round_num"] for i in train_idx})
    print(f"\nFitting prior on rounds {train_rounds} ({len(train_idx)} seeds)...")

    lookup = DirichletLookup()
    lookup.fit(X[train_idx], Y[train_idx], [settlements[i] for i in train_idx])
    print(lookup.summary())

    # ------------------------------------------------------------------
    # 3. Load round data for round_num
    # ------------------------------------------------------------------
    test_idx = sorted(
        [i for i, m in enumerate(meta) if m["round_num"] == round_num],
        key=lambda i: meta[i]["seed_idx"],
    )
    num_seeds = len(test_idx)
    round_id  = meta[test_idx[0]]["round_id"]

    rd_path = DATA_DIR / "rounds" / round_id / "round_data.json"
    if not rd_path.exists():
        print(f"✗ round_data.json not found: {rd_path}")
        return

    round_data           = json.loads(rd_path.read_text())
    terrains, setts_list = parse_initial_states(round_data)
    H, W                 = terrains[0].shape
    ground_truth         = [Y[i] for i in test_idx]    # 8-state for SimulatedSession
    ground_truth_6       = [Y6[i] for i in test_idx]   # 6-class for scoring

    print(f"\nRound {round_num}: {num_seeds} seeds, {H}×{W} map")

    # ------------------------------------------------------------------
    # 4. Build Dirichlet priors
    # ------------------------------------------------------------------
    print(f"\nBuilding priors for {num_seeds} seeds...")
    alphas = [lookup.build_prior(terrains[s], setts_list[s]) for s in range(num_seeds)]

    prior_preds  = [
        apply_floor_and_normalize(
            collapse_8state_to_6class(a / a.sum(axis=-1, keepdims=True))
        )
        for a in alphas
    ]
    prior_scores = [competition_score(ground_truth_6[s], prior_preds[s]) for s in range(num_seeds)]
    for s, sc in enumerate(prior_scores):
        print(f"  Seed {s}: prior score = {sc:.1f}/100")
    print(f"  Prior mean: {np.mean(prior_scores):.1f}/100")

    # ------------------------------------------------------------------
    # 5. Targeted coverage queries
    # ------------------------------------------------------------------
    # Load settlement records if available for offline regime inference
    sett_records = None
    sett_records_path = DATA_DIR / "rounds" / round_id / "settlement_records.json"
    if sett_records_path.exists():
        sett_records = json.loads(sett_records_path.read_text())
        print(f"  Settlement records loaded ({sum(len(v) for v in sett_records.values())} total)")
    else:
        print(f"  No settlement_records.json for this round (regime inference disabled)")

    sim = SimulatedSession(
        ground_truth, budget=budget, rng_seed=rng_seed,
        settlement_records=sett_records,
    )

    if adaptive:
        # ------------------------------------------------------------------
        # 5–6. Adaptive multi-pass query strategy
        # ------------------------------------------------------------------
        observations, extra_obs, settlement_log = run_adaptive_queries(
            sim, num_seeds, terrains, setts_list, alphas,
            budget=budget, max_settlement_dist=8,
            pool_weight=pool_weight, temperature=temperature,
            floor_base=floor_base, floor_obs_scale=floor_obs_scale,
            map_size=(H, W),
        )

        # Mid-point scores (before refinement)
        preds_mid = build_predictions_pooled(
            alphas, observations, terrains, setts_list,
            pool_weight=pool_weight, temperature=temperature,
            floor_base=floor_base, floor_obs_scale=floor_obs_scale,
        )
        mid_scores = [competition_score(ground_truth_6[s], preds_mid[s])
                      for s in range(num_seeds)]

    else:
        # ------------------------------------------------------------------
        # 5. Uniform targeted coverage
        # ------------------------------------------------------------------
        resample_budget = max(5, budget // 10)
        coverage_budget = budget - resample_budget
        vps_per_seed = coverage_budget // num_seeds

        viewports_per_seed = [
            dynamic_coverage_viewports(
                terrains[s], setts_list[s], max_viewports=vps_per_seed,
            )
            for s in range(num_seeds)
        ]
        total_vps = sum(len(vps) for vps in viewports_per_seed)
        for s, vps in enumerate(viewports_per_seed):
            print(f"  Seed {s}: {len(vps)} targeted viewports")

        print(f"\n--- Phase 1: Targeted coverage "
              f"({total_vps} viewports across {num_seeds} seeds) ---")
        observations, _, settlement_log = run_coverage_queries(
            sim, num_seeds, viewports_per_seed, (H, W),
        )

        # Score with pooled predictions (cross-seed feature pooling)
        preds_mid = build_predictions_pooled(
            alphas, observations, terrains, setts_list,
            pool_weight=pool_weight, temperature=temperature,
            floor_base=floor_base, floor_obs_scale=floor_obs_scale,
        )
        mid_scores = [competition_score(ground_truth_6[s], preds_mid[s])
                      for s in range(num_seeds)]
        for s, sc in enumerate(mid_scores):
            gain = sc - prior_scores[s]
            print(f"  Seed {s}: coverage+pooling score = {sc:.1f}/100  "
                  f"({gain:+.1f} vs prior)")
        print(f"  Coverage+pooling mean: {np.mean(mid_scores):.1f}/100  "
              f"(budget remaining: {sim.budget_remaining})")

        # ------------------------------------------------------------------
        # 6. Targeted resampling
        # ------------------------------------------------------------------
        remaining   = sim.budget_remaining
        alphas_post = [alphas[s] + observations[s] for s in range(num_seeds)]

        print(f"\n--- Phase 2: Resampling ({remaining} queries remaining) ---")
        extra_obs, _ = run_resampling_queries(
            sim, num_seeds, terrains, setts_list, remaining,
            alphas_post=alphas_post, map_size=(H, W),
            settlement_log=settlement_log,
        )

    # Regime inference from settlement payloads
    regime = estimate_regime(settlement_log, setts_list)
    print(f"\n  Regime: {regime_summary(regime)}")

    adj = regime_adjustments(regime)
    final_pool_weight = pool_weight * adj["pool_weight_mult"]
    final_floor_base  = floor_base * adj["floor_base_mult"]
    final_temperature = temperature + adj["temperature_adj"]
    if adj != {"pool_weight_mult": 1.0, "floor_base_mult": 1.0, "temperature_adj": 0.0}:
        print(f"  Regime adjustments: pool_weight={final_pool_weight:.2f}, "
              f"floor_base={final_floor_base:.4f}, temperature={final_temperature:.2f}")

    preds_final  = build_predictions_pooled(
        alphas, observations, terrains, setts_list, extra_obs=extra_obs,
        pool_weight=final_pool_weight, temperature=final_temperature,
        floor_base=final_floor_base, floor_obs_scale=floor_obs_scale,
    )

    # Optional blended neural correction (matches live path in submit_solution.py)
    NEURAL_BLEND = 0.40
    if NEURAL_BLEND > 0:
        try:
            from astar.neural.predict import load_predictor
            neural = load_predictor()
        except Exception:
            neural = None
        if neural is not None:
            preds_neural = neural.correct_predictions(
                preds_final, terrains, setts_list, observations, extra_obs,
            )
            preds_blended = [
                (1.0 - NEURAL_BLEND) * b + NEURAL_BLEND * n
                for b, n in zip(preds_final, preds_neural)
            ]
            # Report all three: v3, neural, blend
            v3_scores_here = [competition_score(ground_truth_6[s], preds_final[s]) for s in range(num_seeds)]
            neural_scores = [competition_score(ground_truth_6[s], preds_neural[s]) for s in range(num_seeds)]
            blend_scores = [competition_score(ground_truth_6[s], preds_blended[s]) for s in range(num_seeds)]
            print(f"\n  Neural: v3={np.mean(v3_scores_here):.1f}  "
                  f"neural={np.mean(neural_scores):.1f}  "
                  f"blend({NEURAL_BLEND:.0%})={np.mean(blend_scores):.1f}  "
                  f"Δblend={np.mean(blend_scores)-np.mean(v3_scores_here):+.1f}")
            preds_final = preds_blended

    final_scores = [competition_score(ground_truth_6[s], preds_final[s]) for s in range(num_seeds)]

    print(f"\n--- Results (Round {round_num}, prior from rounds {train_rounds}) ---")
    print(f"  {'Seed':<6}  {'Prior':>7}  {'Coverage':>9}  {'Final':>7}  {'Gain':>6}")
    print(f"  {'-'*46}")
    for s in range(num_seeds):
        gain = final_scores[s] - prior_scores[s]
        print(f"  {s:<6}  {prior_scores[s]:>7.1f}  {mid_scores[s]:>9.1f}  "
              f"{final_scores[s]:>7.1f}  {gain:>+6.1f}")
    print(f"  {'-'*46}")
    print(f"  {'Mean':<6}  {np.mean(prior_scores):>7.1f}  {np.mean(mid_scores):>9.1f}  "
          f"{np.mean(final_scores):>7.1f}  "
          f"{np.mean(final_scores) - np.mean(prior_scores):>+6.1f}")
    print(f"\n  Queries used: {sim.queries_used}/{budget}")

    # ------------------------------------------------------------------
    # 7. Combined per-class heatmap (all seeds in one figure)
    # ------------------------------------------------------------------
    Path("figs").mkdir(exist_ok=True)

    fig_maps, axes_maps = plt.subplots(
        num_seeds * 2, 6, figsize=(22, 8 * num_seeds)
    )
    fig_maps.suptitle(
        f"Round {round_num} — Final predictions vs Ground Truth\n"
        f"Prior trained on rounds {train_rounds}",
        fontsize=14, fontweight="bold",
    )
    for s in range(num_seeds):
        row0 = s * 2
        row1 = row0 + 1
        for c in range(6):
            ax = axes_maps[row0, c]
            im = ax.imshow(preds_final[s][:, :, c], origin="upper",
                           interpolation="nearest", cmap="gnuplot", vmin=0, vmax=1)
            ax.set_title(f"Pred P({CLASS_NAMES[c]})", fontsize=8,
                         color=CLASS_COLORS[c] * 0.7)
            ax.axis("off")
            plt.colorbar(im, ax=ax, shrink=0.6, format="%.2f")

            ax2 = axes_maps[row1, c]
            im2 = ax2.imshow(ground_truth_6[s][:, :, c], origin="upper",
                             interpolation="nearest", cmap="gnuplot", vmin=0, vmax=1)
            ax2.set_title(f"GT P({CLASS_NAMES[c]})", fontsize=8)
            ax2.axis("off")
            plt.colorbar(im2, ax=ax2, shrink=0.6, format="%.2f")

        seed_label = (
            f"Seed {s}\n{final_scores[s]:.1f}/100\n"
            f"(prior {prior_scores[s]:.1f})"
        )
        axes_maps[row0, 0].text(
            -0.12, 0.5, seed_label,
            transform=axes_maps[row0, 0].transAxes,
            fontsize=9, fontweight="bold", va="center", rotation=90,
        )

    plt.tight_layout()
    heatmap_path = f"figs/main_r{round_num}_heatmaps.png"
    plt.savefig(heatmap_path, dpi=100, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close(fig_maps)

    # ------------------------------------------------------------------
    # 8. Score progression bar chart
    # ------------------------------------------------------------------
    fig_scores, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(num_seeds)
    w = 0.25
    bars_prior = ax.bar(x - w, prior_scores, w, label="Prior",    color="steelblue")
    bars_mid   = ax.bar(x,     mid_scores,   w, label="Coverage", color="darkorange")
    bars_final = ax.bar(x + w, final_scores, w, label="Final",    color="seagreen")

    for bars in (bars_prior, bars_mid, bars_final):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in range(num_seeds)])
    ax.set_ylabel("Competition score (0–100, higher is better)")
    ax.set_title(
        f"Round {round_num} — Score progression\n"
        f"Prior trained on rounds {train_rounds}"
    )
    ax.set_ylim(0, 105)
    ax.legend()
    mean_final = np.mean(final_scores)
    ax.axhline(mean_final, color="seagreen", linestyle="--", alpha=0.6)
    ax.text(num_seeds - 0.4, mean_final + 1,
            f"Mean: {mean_final:.1f}", color="seagreen", fontsize=9)

    plt.tight_layout()
    scores_path = f"figs/main_r{round_num}_scores.png"
    plt.savefig(scores_path, dpi=150, bbox_inches="tight")
    if show_fig:
        plt.show()
    else:
        plt.close(fig_scores)

    print(f"\nFigures saved to {heatmap_path}, {scores_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round",  type=int, default=ROUND_NUM,
                        help=f"Round number to replay (default: {ROUND_NUM})")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help=f"Query budget (default: {BUDGET})")
    parser.add_argument("--seed",   type=int, default=RNG_SEED,
                        help=f"RNG seed for SimulatedSession (default: {RNG_SEED})")
    parser.add_argument("--show-fig", action="store_true", default=SHOW_FIG,
                        help="Display figures interactively (default: False)")
    parser.add_argument("--pool-weight", type=float, default=POOL_WEIGHT,
                        help=f"Cross-seed pooling strength (default: {POOL_WEIGHT})")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help=f"Posterior temperature (default: {TEMPERATURE})")
    parser.add_argument("--floor-base", type=float, default=FLOOR_BASE,
                        help=f"Dynamic floor base (default: {FLOOR_BASE})")
    parser.add_argument("--floor-obs-scale", type=float, default=FLOOR_OBS_SCALE,
                        help=f"Dynamic floor observation scale (default: {FLOOR_OBS_SCALE})")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive multi-pass query strategy")
    args = parser.parse_args()
    main(
        args.round, args.budget, args.seed,
        show_fig=args.show_fig,
        pool_weight=args.pool_weight,
        temperature=args.temperature,
        floor_base=args.floor_base,
        floor_obs_scale=args.floor_obs_scale,
        adaptive=args.adaptive,
    )
