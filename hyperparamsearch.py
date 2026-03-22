#!/usr/bin/env python3
"""
Bayesian hyperparameter search — hyperparamsearch.py

Uses Optuna (multivariate TPE sampler) with multi-process parallelism and a
SQLite-backed study for persistence. Safe to interrupt and resume: all trials
are written to the database as they complete, so --resume picks up exactly
where the run left off.

Usage:
    python hyperparamsearch.py                        # 500 trials, all CPUs
    python hyperparamsearch.py --rounds 10 11 12 13   # specific rounds
    python hyperparamsearch.py --n-trials 200         # fewer trials
    python hyperparamsearch.py --n-jobs 5             # limit to 5 workers
    python hyperparamsearch.py --resume               # resume existing study
    python hyperparamsearch.py --adaptive             # adaptive query strategy
"""

# ── USER CONFIG ─────────────────────────────────────────────────────────────
TEST_ROUNDS  = [10, 11, 12, 13, 14]   # rounds to evaluate on
RNG_SEED     = 42                      # reproducibility
BUDGET       = 50                      # query budget per round
N_TRIALS     = 500                     # total Bayesian trials to run
STUDY_NAME   = "astar_search"
DB_PATH      = "data/hyperparamsearch.db"      # SQLite checkpoint (safe to resume)
RESULTS_PATH = "data/hyperparamsearch_results.json"   # human-readable JSON export
# ────────────────────────────────────────────────────────────────────────────

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from astar.data       import parse_initial_states
from astar.dirichlet  import DirichletLookup
from astar.evaluation import load_dataset
from astar.inference  import (
    build_predictions_pooled,
    dynamic_coverage_viewports,
    run_adaptive_queries,
    run_coverage_queries,
    run_resampling_queries,
)
from astar.scoring import competition_score
from astar.session import SimulatedSession

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Parameter search space
# ---------------------------------------------------------------------------

def suggest_params(trial: optuna.Trial) -> dict:
    """Sample one hyperparameter configuration using TPE.

    Ranges are centred on the best-found values from grid search (rounds 10-14):
      tau=10, n_reg=5, kappa_min=2, epsilon=0.01, pool_weight=1.0,
      max_settlement_dist=12, temperature=1.0, floor_base=0.001, floor_obs_scale=0.02
    """
    return {
        # ── Prior regularisation ──────────────────────────────────────────
        # Shrinkage of fine-bin means toward coarser parent bins.
        # Best=10. Log-scale: explore from very sharp (2) to very smooth (100).
        "tau": trial.suggest_float("tau", 2.0, 100.0, log=True),

        # Shrinkage of per-bin κ toward terrain-group defaults.
        # Best=5. Log-scale: from trust-empirical (1) to heavy-prior (100).
        "n_reg": trial.suggest_float("n_reg", 1.0, 100.0, log=True),

        # Hard floor on Dirichlet concentration κ.
        # Best=2. Rarely binding; narrow search.
        "kappa_min": trial.suggest_float("kappa_min", 1.0, 8.0),

        # Pseudo-count per class — insensitive parameter.
        # Best=0.01. Log-scale: prevents zero-probability edge cases.
        "epsilon": trial.suggest_float("epsilon", 0.001, 0.05, log=True),

        # ── Observation breadth & pooling ────────────────────────────────
        # Cross-seed pooling multiplier. Best=1.0.
        # Log-scale: ranges from under-pooling (0.3) to aggressive pooling (5.0).
        "pool_weight": trial.suggest_float("pool_weight", 0.3, 5.0, log=True),

        # Manhattan distance cutoff for dynamic cells.
        # Best=12. Integer: narrow search around known-good value.
        "max_settlement_dist": trial.suggest_int("max_settlement_dist", 6, 24),

        # ── Posterior calibration ─────────────────────────────────────────
        # Temperature: T<1 sharpens predictions, T>1 softens. Best=1.0.
        "temperature": trial.suggest_float("temperature", 0.7, 1.3),

        # Minimum probability floor for well-observed cells. Best=0.001.
        # Log-scale: too small risks zero-prob blow-ups; too large over-smooths.
        "floor_base": trial.suggest_float("floor_base", 0.0002, 0.01, log=True),

        # Additional floor for unobserved cells. Best=0.02.
        # Log-scale: decays with observation count.
        "floor_obs_scale": trial.suggest_float("floor_obs_scale", 0.005, 0.1, log=True),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_config(
    cfg: dict,
    dataset: dict,
    test_rounds: list[int],
    budget: int,
    rng_seed: int,
    adaptive: bool = False,
) -> dict:
    """Run the full pipeline for one config on all test rounds.

    Returns dict with per-round scores and mean score.
    """
    meta        = dataset["meta"]
    X           = dataset["X"]
    Y           = dataset["Y"]       # 8-state for training/simulation
    Y6          = dataset["Y6"]      # 6-class for scoring
    settlements = dataset["settlements"]

    per_round = {}

    for round_num in test_rounds:
        train_idx = [i for i, m in enumerate(meta) if m["round_num"] < round_num]
        if not train_idx:
            continue

        lookup = DirichletLookup(
            tau=cfg["tau"],
            n_reg=cfg["n_reg"],
            kappa_min=cfg["kappa_min"],
            epsilon=cfg["epsilon"],
        )
        lookup.fit(X[train_idx], Y[train_idx], [settlements[i] for i in train_idx])

        test_idx = sorted(
            [i for i, m in enumerate(meta) if m["round_num"] == round_num],
            key=lambda i: meta[i]["seed_idx"],
        )
        if not test_idx:
            continue

        round_id = meta[test_idx[0]]["round_id"]
        rd_path  = DATA_DIR / "rounds" / round_id / "round_data.json"
        if not rd_path.exists():
            continue

        round_data           = json.loads(rd_path.read_text())
        terrains, setts_list = parse_initial_states(round_data)
        H, W                 = terrains[0].shape
        ground_truth         = [Y[i] for i in test_idx]     # 8-state for SimulatedSession
        ground_truth_6       = [Y6[i] for i in test_idx]   # 6-class for scoring
        num_seeds            = len(test_idx)

        alphas = [lookup.build_prior(terrains[s], setts_list[s]) for s in range(num_seeds)]

        sett_records_path = DATA_DIR / "rounds" / round_id / "settlement_records.json"
        sett_records = (json.loads(sett_records_path.read_text())
                        if sett_records_path.exists() else None)

        sim = SimulatedSession(ground_truth, budget=budget, rng_seed=rng_seed,
                               settlement_records=sett_records)

        if adaptive:
            observations, extra_obs, _ = run_adaptive_queries(
                sim, num_seeds, terrains, setts_list, alphas,
                budget=budget,
                max_settlement_dist=cfg["max_settlement_dist"],
                pool_weight=cfg["pool_weight"],
                temperature=cfg["temperature"],
                floor_base=cfg["floor_base"],
                floor_obs_scale=cfg["floor_obs_scale"],
                map_size=(H, W),
                verbose=False,
            )
        else:
            resample_budget = max(5, budget // 10)
            vps_per_seed    = (budget - resample_budget) // num_seeds

            viewports_per_seed = [
                dynamic_coverage_viewports(
                    terrains[s], setts_list[s],
                    max_viewports=vps_per_seed,
                    max_settlement_dist=cfg["max_settlement_dist"],
                )
                for s in range(num_seeds)
            ]
            observations, _, _ = run_coverage_queries(
                sim, num_seeds, viewports_per_seed, (H, W), verbose=False,
            )
            remaining   = sim.budget_remaining
            alphas_post = [alphas[s] + observations[s] for s in range(num_seeds)]
            extra_obs, _ = run_resampling_queries(
                sim, num_seeds, terrains, setts_list, remaining,
                alphas_post=alphas_post, map_size=(H, W), verbose=False,
            )

        preds = build_predictions_pooled(
            alphas, observations, terrains, setts_list,
            extra_obs=extra_obs,
            pool_weight=cfg["pool_weight"],
            temperature=cfg["temperature"],
            floor_base=cfg["floor_base"],
            floor_obs_scale=cfg["floor_obs_scale"],
        )

        scores = [competition_score(ground_truth_6[s], preds[s]) for s in range(num_seeds)]
        per_round[round_num] = {
            "scores": scores,
            "mean":   float(np.mean(scores)),
        }

    all_means  = [v["mean"] for v in per_round.values()]
    mean_score = float(np.mean(all_means)) if all_means else 0.0
    return {"per_round": per_round, "mean_score": mean_score}


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker(
    study_name: str,
    storage_url: str,
    n_trials: int,
    test_rounds: list,
    budget: int,
    rng_seed: int,
    adaptive: bool,
    worker_id: int,
    results_path: str,
    save_every: int,
) -> None:
    """Runs in a separate process. Loads dataset, then optimises n_trials trials."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Add small jitter so workers don't all hit the DB at exactly the same time
    time.sleep(worker_id * 0.3)

    dataset = load_dataset(DATA_DIR)

    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_params(trial)
        result = evaluate_config(cfg, dataset, test_rounds, budget, rng_seed, adaptive)
        # Store per-round breakdown as user attributes for later inspection
        for rn, rv in result["per_round"].items():
            trial.set_user_attr(f"R{rn}_mean", rv["mean"])
        return result["mean_score"]

    def on_trial_complete(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        round_str = "  ".join(
            f"R{rn}={trial.user_attrs.get(f'R{rn}_mean', 0):.1f}"
            for rn in test_rounds
        )
        print(
            f"  [W{worker_id:02d}] #{n_done:>4}  "
            f"score={trial.value:5.2f}  best={study.best_value:5.2f}  "
            f"{round_str}",
            flush=True,
        )
        # Periodically export JSON so results are human-readable during the run
        if n_done % save_every == 0:
            _save_results_json(study, results_path)

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[on_trial_complete],
        show_progress_bar=False,
        catch=(Exception,),   # failed trials are marked FAIL and skipped
    )


# ---------------------------------------------------------------------------
# Results helpers
# ---------------------------------------------------------------------------

def _save_results_json(study: optuna.Study, path: str) -> None:
    """Export completed trials to JSON sorted by score descending."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value or 0, reverse=True)
    results = [
        {"trial_id": t.number, "mean_score": t.value, "config": t.params,
         "per_round": {k: v for k, v in t.user_attrs.items()}}
        for t in completed
    ]
    Path(path).parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _print_top_results(study: optuna.Study, test_rounds: list, top_n: int = 10) -> None:
    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value or 0, reverse=True,
    )
    defaults = {
        "tau": 10.0, "n_reg": 5.0, "kappa_min": 2.0, "epsilon": 0.01,
        "pool_weight": 1.0, "max_settlement_dist": 12,
        "temperature": 1.0, "floor_base": 0.001, "floor_obs_scale": 0.02,
    }
    print(f"\n{'='*72}")
    print(f"Top {top_n} configurations  (mean over rounds {test_rounds})")
    print(f"{'='*72}")
    for rank, t in enumerate(completed[:top_n], 1):
        changed = {
            k: v for k, v in t.params.items()
            if abs(v - defaults.get(k, v)) / (abs(defaults.get(k, v)) + 1e-9) > 0.05
        }
        round_str = "  ".join(
            f"R{rn}={t.user_attrs.get(f'R{rn}_mean', 0):.1f}" for rn in test_rounds
        )
        print(f"  {rank:>3}.  score={t.value:5.2f}  {round_str}  {changed}")

    best = study.best_trial
    print(f"\nBest config (trial #{best.number},  score={best.value:.2f}/100):")
    for k, v in best.params.items():
        marker = "  ← changed" if abs(v - defaults.get(k, v)) / (abs(defaults.get(k, v)) + 1e-9) > 0.05 else ""
        print(f"  {k:25s} = {v:.4g}{marker}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rounds",     type=int, nargs="+", default=TEST_ROUNDS)
    parser.add_argument("--n-trials",   type=int, default=N_TRIALS,
                        help=f"Total Bayesian trials (default: {N_TRIALS})")
    parser.add_argument("--n-jobs",     type=int, default=None,
                        help="Parallel worker processes (default: all CPUs)")
    parser.add_argument("--budget",     type=int, default=BUDGET)
    parser.add_argument("--seed",       type=int, default=RNG_SEED)
    parser.add_argument("--study",      type=str, default=STUDY_NAME)
    parser.add_argument("--db",         type=str, default=DB_PATH,
                        help="SQLite path for persistent study storage")
    parser.add_argument("--out",        type=str, default=RESULTS_PATH)
    parser.add_argument("--resume",     action="store_true",
                        help="Resume an existing study (auto-detected if DB exists)")
    parser.add_argument("--adaptive",   action="store_true",
                        help="Use adaptive multi-pass query strategy")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save JSON results every N completed trials per worker")
    args = parser.parse_args()

    n_jobs = args.n_jobs or multiprocessing.cpu_count()
    storage_url = f"sqlite:///{args.db}"
    strategy    = "adaptive" if args.adaptive else "uniform"

    Path(args.db).parent.mkdir(exist_ok=True)

    # Create study (idempotent: load_if_exists=True covers --resume)
    study = optuna.create_study(
        study_name=args.study,
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        load_if_exists=True,
    )
    existing = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    print(f"Bayesian hyperparameter search  [{strategy}]")
    print(f"  Rounds:     {args.rounds}")
    print(f"  Trials:     {args.n_trials}  ({existing} already completed)")
    print(f"  Workers:    {n_jobs}")
    print(f"  DB:         {args.db}  ← safe to interrupt; resume with --resume")
    print(f"  Results:    {args.out}")
    if existing:
        print(f"  Best so far: {study.best_value:.2f}/100")
    print()

    # Distribute trials evenly across workers
    base, extra = divmod(args.n_trials, n_jobs)
    worker_trials = [base + (1 if i < extra else 0) for i in range(n_jobs)]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                _worker,
                args.study, storage_url, worker_trials[i],
                args.rounds, args.budget, args.seed, args.adaptive,
                i, args.out, args.save_every,
            ): i
            for i in range(n_jobs)
        }
        for future in as_completed(futures):
            wid = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  [W{wid:02d}] worker crashed: {e}")

    elapsed = time.time() - t0
    print(f"\nSearch complete in {elapsed/60:.1f} min")

    # Final save and summary
    study = optuna.load_study(study_name=args.study, storage=storage_url)
    _save_results_json(study, args.out)
    print(f"Results saved to {args.out}  ({len(study.trials)} total trials)")
    _print_top_results(study, args.rounds)


if __name__ == "__main__":
    main()
