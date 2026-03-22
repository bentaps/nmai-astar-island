"""
Dirichlet prior construction for Astar Island.

All priors are data-driven via DirichletLookup (astar.dirichlet).
There are no hand-tuned fallbacks — a fitted lookup table is required.
Run evaluate.py to (re-)fit it whenever new round data is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import NUM_CLASSES

_DEFAULT_LOOKUP_PATH = Path(__file__).parent.parent / "data" / "dirichlet_lookup.json"


def load_lookup(path: str | Path = _DEFAULT_LOOKUP_PATH):
    """Load the fitted Dirichlet lookup table.

    Args:
        path: path to dirichlet_lookup.json.

    Returns:
        DirichletLookup instance.

    Raises:
        FileNotFoundError / ValueError if the file is missing or corrupt.
    """
    from .dirichlet import DirichletLookup
    return DirichletLookup.load(path)


def build_prior(
    terrain:     np.ndarray,
    settlements: list[dict],
    lookup,
) -> np.ndarray:
    """Build (H, W, 6) Dirichlet α prior for one seed.

    Args:
        terrain:     (H, W) terrain code grid.
        settlements: list of settlement dicts with 'x', 'y' keys.
        lookup:      fitted DirichletLookup (required).

    Returns:
        (H, W, NUM_CLASSES) Dirichlet pseudo-count array.
    """
    if lookup is None:
        raise ValueError(
            "No lookup table provided. Run evaluate.py to fit one, "
            "then load it with load_lookup()."
        )
    return lookup.build_prior(terrain, settlements)


def build_uniform_baseline(H: int, W: int, num_seeds: int) -> list[np.ndarray]:
    """Uniform 1/6 baseline — only for completely unseen round types."""
    return [np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES) for _ in range(num_seeds)]
