"""
Evaluation utilities: leave-one-round-out cross-validation and reporting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dirichlet import DirichletLookup
from .features import NUM_CLASSES, NUM_RAW_CODES, TERRAIN_TO_CLASS, expand_6class_to_8state, collapse_8state_to_6class
from .scoring import apply_floor_and_normalize, entropy_weighted_kl


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_dir: str | Path = "data") -> dict:
    """Load the historical training dataset.

    Expects:
        data/X_initial.npy       — (N, H, W) initial terrain codes
        data/Y_ground_truth.npy  — (N, H, W, 6) ground truth distributions
        data/training_meta.json  — list of {round_num, round_id, seed_idx, score}

    Also loads per-round settlement data from data/rounds/{round_id}/round_data.json.

    Returns:
        dict with keys:
            X, Y, meta, settlements, round_numbers
    """
    data_dir = Path(data_dir)
    X    = np.load(data_dir / "X_initial.npy")
    Y    = np.load(data_dir / "Y_ground_truth.npy")
    meta = json.loads((data_dir / "training_meta.json").read_text())

    settlements = []
    for m in meta:
        rd_path = data_dir / "rounds" / m["round_id"] / "round_data.json"
        if rd_path.exists():
            rd       = json.loads(rd_path.read_text())
            si       = m["seed_idx"]
            states   = rd.get("initial_states", [])
            settlements.append(states[si].get("settlements", []) if si < len(states) else [])
        else:
            settlements.append([])

    # Expand ground truth from 6-class to 8-state using initial terrain
    Y8 = expand_6class_to_8state(Y, X)

    return {
        "X":             X,
        "Y":             Y8,
        "Y6":            Y,    # original 6-class for scoring
        "meta":          meta,
        "settlements":   settlements,
        "round_numbers": sorted({m["round_num"] for m in meta}),
    }


# ---------------------------------------------------------------------------
# Dataset rebuilding
# ---------------------------------------------------------------------------

def rebuild_training_arrays(data_dir: str | Path = "data") -> int:
    """Rebuild X_initial.npy, Y_ground_truth.npy, training_meta.json from raw JSON.

    Scans data/rounds/*/round_data.json + analysis_seed{i}.json and stacks
    all seeds that have both files into numpy arrays.

    Args:
        data_dir: root data directory (contains X_initial.npy etc.).

    Returns:
        Number of samples written.
    """
    data_dir   = Path(data_dir)
    rounds_dir = data_dir / "rounds"

    X_list, Y_list, meta_list = [], [], []

    for round_dir in sorted(rounds_dir.iterdir()):
        rd_path = round_dir / "round_data.json"
        if not rd_path.exists():
            continue

        rd         = json.loads(rd_path.read_text())
        round_num  = rd.get("round_number")
        round_id   = rd.get("id") or round_dir.name
        states     = rd.get("initial_states", [])

        for seed_idx, state in enumerate(states):
            analysis_path = round_dir / f"analysis_seed{seed_idx}.json"
            if not analysis_path.exists():
                continue

            analysis = json.loads(analysis_path.read_text())
            gt = analysis.get("ground_truth")
            if gt is None:
                continue

            X_list.append(np.array(state["grid"], dtype=int))
            Y_list.append(np.array(gt, dtype=float))
            meta_list.append({
                "round_num":  round_num,
                "round_id":   round_id,
                "seed_idx":   seed_idx,
                "score":      analysis.get("score"),
            })

    if not X_list:
        return 0

    np.save(data_dir / "X_initial.npy",      np.array(X_list))
    np.save(data_dir / "Y_ground_truth.npy", np.array(Y_list))
    (data_dir / "training_meta.json").write_text(json.dumps(meta_list, indent=2))

    return len(X_list)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def leave_one_round_out_cv(
    dataset:      dict,
    kappa_min:    float = 2.0,
    kappa_max:    float = 200.0,
    epsilon:      float = 0.01,
    tau:          float = 10.0,
    n_reg:        float = 50.0,
    prob_floor:   float = 0.01,
    train_rounds: list[int] | None = None,
    test_rounds:  list[int] | None = None,
) -> dict:
    """Leave-one-round-out cross-validation of DirichletLookup.

    Returns:
        dict with per_round scores, mean_score, and per-seed breakdowns.
    """
    X            = dataset["X"]
    Y            = dataset["Y"]       # 8-state for training
    Y6           = dataset["Y6"]      # 6-class for scoring
    meta         = dataset["meta"]
    settlements  = dataset["settlements"]
    round_numbers = dataset["round_numbers"]

    if train_rounds is not None and test_rounds is not None:
        folds = [(train_rounds, test_rounds)]
    else:
        folds = [
            ([r for r in round_numbers if r != held], [held])
            for held in round_numbers
        ]

    per_round = {}

    for train_rns, test_rns in folds:
        train_idx = [i for i, m in enumerate(meta) if m["round_num"] in train_rns]
        test_idx  = [i for i, m in enumerate(meta) if m["round_num"] in test_rns]
        if not train_idx or not test_idx:
            continue

        lookup = DirichletLookup(
            kappa_min=kappa_min, kappa_max=kappa_max,
            epsilon=epsilon, tau=tau, n_reg=n_reg,
        )
        lookup.fit(X[train_idx], Y[train_idx],
                   [settlements[i] for i in train_idx])

        for test_round in test_rns:
            round_idx = [i for i in test_idx if meta[i]["round_num"] == test_round]
            scores = []
            for i in round_idx:
                alpha = lookup.build_prior(X[i], settlements[i])
                q8    = alpha / alpha.sum(axis=-1, keepdims=True)
                q     = apply_floor_and_normalize(
                    collapse_8state_to_6class(q8), floor=prob_floor
                )
                scores.append(entropy_weighted_kl(Y6[i], q))

            per_round[test_round] = {
                "score":   float(np.mean(scores)),
                "scores":  scores,
                "n_seeds": len(round_idx),
            }

    all_scores   = [v["score"] for v in per_round.values()]
    mean_score   = float(np.mean(all_scores)) if all_scores else 0.0

    return {
        "per_round":  per_round,
        "mean_score": mean_score,
    }


def format_cv_results(results: dict) -> str:
    """Format CV results as a readable string."""
    lines = [
        "=" * 55,
        "Leave-One-Round-Out Cross-Validation (DirichletLookup)",
        "=" * 55,
        "",
        f"{'Round':>6}  {'Score':>10}  {'Seeds':>5}",
        "-" * 55,
    ]
    for rn in sorted(results["per_round"]):
        r = results["per_round"][rn]
        lines.append(f"  R{rn:>3}  {r['score']:>10.2f}  {r['n_seeds']:>5}")
        for j, s in enumerate(r["scores"]):
            lines.append(f"         seed {j}: {s:>8.2f}")
    lines.extend([
        "-" * 55,
        f"  Mean   {results['mean_score']:>10.2f}",
        "",
        "Score = competition score 0–100 (higher is better).",
        "=" * 55,
    ])
    return "\n".join(lines)
