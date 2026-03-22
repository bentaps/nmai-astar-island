"""
Inference integration: apply trained residual model on top of v3 baseline.

Usage:
    from astar.neural.predict import NeuralPredictor

    predictor = NeuralPredictor("data/neural_model.pt")
    corrected = predictor.correct_predictions(
        v3_preds, terrains, settlements_list, observations, extra_obs,
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .episodes import build_feature_planes, build_round_latent_features
from .model import ResidualNeuralProcess
from ..features import NUM_RAW_CODES
from ..scoring import apply_floor_and_normalize


class NeuralPredictor:
    """Apply trained residual neural process to correct v3 predictions."""

    def __init__(
        self,
        model_path: str | Path = "data/neural_model.pt",
        device_name: str = "auto",
        floor: float = 0.0002143,
    ):
        # Avoid torch threading issues on some machines
        torch.set_num_threads(1)

        if device_name == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_name)

        self.floor = floor

        # Load checkpoint and reconstruct model from saved kwargs if available
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model_kwargs = ckpt.get("model_kwargs", {
            "in_channels": 27,
            "base_channels": 32,
            "round_latent_dim": 16,
            "out_channels": 6,
        })
        self.model = ResidualNeuralProcess(**model_kwargs)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def correct_predictions(
        self,
        v3_preds: list[np.ndarray],
        terrains: list[np.ndarray],
        settlements_list: list[list[dict]],
        observations: dict[int, np.ndarray],
        extra_obs: dict[int, np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """Apply neural corrections to v3 baseline predictions.

        Args:
            v3_preds: list of (H, W, 6) v3 posterior predictions per seed.
            terrains: list of (H, W) terrain grids per seed.
            settlements_list: list of settlement lists per seed.
            observations: dict[seed_idx] -> (H, W, 8) observation counts.
            extra_obs: optional dict[seed_idx] -> (H, W, 8) extra counts.

        Returns:
            list of (H, W, 6) corrected probability tensors per seed.
        """
        num_seeds = len(v3_preds)

        # Build features
        total_obs = {}
        for s in range(num_seeds):
            obs = observations.get(s, np.zeros((*terrains[0].shape, NUM_RAW_CODES)))
            if extra_obs and s in extra_obs:
                obs = obs + extra_obs[s]
            total_obs[s] = obs

        # Build feature planes
        features_list = []
        for s in range(num_seeds):
            feat = build_feature_planes(
                terrains[s], settlements_list[s], total_obs[s], v3_preds[s],
            )
            # (H, W, 27) -> (1, 27, H, W)
            feat_t = torch.from_numpy(
                feat.astype(np.float32).transpose(2, 0, 1)
            ).unsqueeze(0).to(self.device)
            features_list.append(feat_t)

        # V3 predictions as tensors
        v3_list = []
        for s in range(num_seeds):
            v3_t = torch.from_numpy(
                v3_preds[s].astype(np.float32).transpose(2, 0, 1)
            ).unsqueeze(0).to(self.device)
            v3_list.append(v3_t)

        # Forward pass with seed pooling
        preds = self.model(features_list, v3_list)

        # Convert back to numpy
        corrected = []
        for pred in preds:
            q = pred.squeeze(0).cpu().numpy()  # (6, H, W)
            q = q.transpose(1, 2, 0)           # (H, W, 6)
            q = apply_floor_and_normalize(q, floor=self.floor)
            corrected.append(q)

        return corrected


def load_predictor(
    model_path: str | Path = "data/neural_model.pt",
) -> NeuralPredictor | None:
    """Load a trained predictor, or return None if model doesn't exist."""
    if not Path(model_path).exists():
        return None
    try:
        return NeuralPredictor(model_path)
    except Exception as e:
        print(f"Warning: could not load neural model: {e}")
        return None
