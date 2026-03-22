#!/usr/bin/env python3
"""
Evaluate the data-driven Dirichlet prior against the hand-tuned baseline.

Usage:
    python evaluate.py                  # Full leave-one-round-out CV
    python evaluate.py --test-round 6   # Train on rounds 1-5, test on round 6

Prints comparison scores and saves the fitted lookup table.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from astar.dirichlet import DirichletLookup
from astar.evaluation import (
    format_cv_results,
    leave_one_round_out_cv,
    load_dataset,
)
from astar.scoring import apply_floor_and_normalize, competition_score, entropy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-round", type=int, default=None,
        help="Hold out this round for testing (default: full leave-one-round-out CV)."
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to data directory."
    )
    parser.add_argument(
        "--save-lookup", type=str, default="data/dirichlet_lookup.json",
        help="Path to save the fitted lookup table."
    )
    args = parser.parse_args()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.data_dir)
    print(
        f"  {len(dataset['X'])} samples, "
        f"{len(dataset['round_numbers'])} rounds: {dataset['round_numbers']}"
    )

    # Run CV
    if args.test_round is not None:
        train_rounds = [r for r in dataset["round_numbers"] if r < args.test_round]
        test_rounds = [args.test_round]
        print(f"\nTrain on rounds {train_rounds}, test on round {args.test_round}")
    else:
        train_rounds = None
        test_rounds = None
        print("\nFull leave-one-round-out cross-validation")

    results = leave_one_round_out_cv(
        dataset,
        train_rounds=train_rounds,
        test_rounds=test_rounds,
    )
    print()
    print(format_cv_results(results))

    # Fit on all training data and save
    if args.test_round is not None:
        train_idx = [
            i for i, m in enumerate(dataset["meta"])
            if m["round_num"] < args.test_round
        ]
    else:
        train_idx = list(range(len(dataset["X"])))

    print(f"\nFitting final lookup table on {len(train_idx)} training samples...")
    lookup = DirichletLookup()
    lookup.fit(
        dataset["X"][train_idx],
        dataset["Y"][train_idx],
        [dataset["settlements"][i] for i in train_idx],
    )
    print(lookup.summary())

    # Save
    lookup.save(args.save_lookup)
    print(f"\nLookup table saved to {args.save_lookup}")

    # Detailed per-cell analysis for test round (if specified)
    if args.test_round is not None:
        print(f"\n{'='*65}")
        print(f"Detailed analysis: Round {args.test_round}")
        print(f"{'='*65}")

        test_idx = [
            i for i, m in enumerate(dataset["meta"])
            if m["round_num"] == args.test_round
        ]

        for i in test_idx:
            m = dataset["meta"][i]
            terrain = dataset["X"][i]
            gt = dataset["Y"][i]
            setts = dataset["settlements"][i]

            alpha = lookup.build_prior(terrain, setts)
            q     = apply_floor_and_normalize(
                alpha / alpha.sum(axis=-1, keepdims=True), floor=0.01
            )

            score    = competition_score(gt, q)
            gt_ent   = entropy(gt)
            mean_ent = float(gt_ent.mean())
            dynamic_cells = int((gt_ent > 0.1).sum())

            print(f"\n  Seed {m['seed_idx']}: score={score:.1f}/100, "
                  f"mean_entropy={mean_ent:.3f}, dynamic_cells={dynamic_cells}")

            # Show α₀ distribution
            alpha_total = alpha.sum(axis=-1)
            print(f"    α₀ stats: min={alpha_total.min():.1f}, "
                  f"median={np.median(alpha_total):.1f}, "
                  f"max={alpha_total.max():.1f}")


if __name__ == "__main__":
    main()
