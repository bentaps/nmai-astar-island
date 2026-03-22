"""
Training loop for the residual neural process.

Uses the episode generator to create training tasks, then trains the
model to predict corrections to v3 baseline that improve competition score.

Loss: entropy-weighted KL divergence (same metric as competition scoring).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from .episodes import EpisodeGenerator
from .model import ResidualNeuralProcess


def entropy_weighted_kl_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Competition-score-aligned loss: entropy-weighted KL divergence.

    Args:
        pred: (B, 6, H, W) predicted probabilities.
        target: (B, 6, H, W) ground truth probabilities.

    Returns:
        Scalar loss (lower is better prediction, higher competition score).
    """
    # Ensure valid distributions
    pred_safe = torch.clamp(pred, min=eps, max=1.0)
    target_safe = torch.clamp(target, min=eps, max=1.0)

    # Entropy of ground truth: H(p) = -Σ p log p
    h = -torch.sum(target_safe * torch.log(target_safe), dim=1)  # (B, H, W)

    # KL divergence: KL(p || q) = Σ p log(p/q)
    kl = torch.sum(target_safe * torch.log(target_safe / pred_safe), dim=1)  # (B, H, W)

    # Entropy-weighted KL: Σ H(p) * KL(p||q) / Σ H(p)
    numerator = (h * kl).sum(dim=(1, 2))   # (B,)
    denominator = h.sum(dim=(1, 2))         # (B,)

    # Avoid division by zero
    valid = denominator > eps
    loss = torch.where(valid, numerator / denominator, torch.zeros_like(numerator))
    return loss.mean()


def competition_score_from_loss(loss_val: float) -> float:
    """Convert entropy-weighted KL loss to approximate competition score."""
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * loss_val))))


def episodes_to_tensors(
    episodes: list[dict],
    device: torch.device,
) -> list[dict]:
    """Convert episode dicts to torch tensors.

    Each episode becomes a list of per-seed tensor dicts.
    """
    tensor_episodes = []
    for ep in episodes:
        seeds = []
        for s in range(ep["num_seeds"]):
            feat = torch.from_numpy(
                ep["features_per_seed"][s].astype(np.float32).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)  # (1, 27, H, W)

            v3 = torch.from_numpy(
                ep["v3_preds_per_seed"][s].astype(np.float32).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)  # (1, 6, H, W)

            target = torch.from_numpy(
                ep["targets_per_seed"][s].astype(np.float32).transpose(2, 0, 1)
            ).unsqueeze(0).to(device)  # (1, 6, H, W)

            seeds.append({
                "features": feat,
                "v3_pred": v3,
                "target": target,
            })
        tensor_episodes.append({
            "seeds": seeds,
            "round_num": ep["round_num"],
        })
    return tensor_episodes


def augment_episode(
    seeds: list[dict],
    rng: np.random.Generator,
) -> list[dict]:
    """Random spatial augmentation: flip + rotation (8 transforms total)."""
    flip_h = rng.random() > 0.5
    flip_v = rng.random() > 0.5
    rot90 = rng.integers(0, 4)  # 0, 1, 2, or 3 quarter-turns

    augmented = []
    for s in seeds:
        feat = s["features"]    # (1, C, H, W)
        v3 = s["v3_pred"]       # (1, 6, H, W)
        target = s["target"]    # (1, 6, H, W)

        if flip_h:
            feat = torch.flip(feat, [3])
            v3 = torch.flip(v3, [3])
            target = torch.flip(target, [3])
        if flip_v:
            feat = torch.flip(feat, [2])
            v3 = torch.flip(v3, [2])
            target = torch.flip(target, [2])
        if rot90 > 0:
            feat = torch.rot90(feat, rot90, [2, 3])
            v3 = torch.rot90(v3, rot90, [2, 3])
            target = torch.rot90(target, rot90, [2, 3])

        augmented.append({
            "features": feat,
            "v3_pred": v3,
            "target": target,
        })
    return augmented


def train_model(
    data_dir: str | Path = "data",
    model_save_path: str | Path = "data/neural_model.pt",
    n_epochs: int = 200,
    episodes_per_round: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    val_rounds: list[int] | None = None,
    device_name: str = "auto",
    budget: int = 50,
) -> dict:
    """Train the residual neural process.

    Uses leave-one-round-out: trains on all rounds except val_rounds,
    validates on val_rounds.

    Args:
        data_dir: path to data directory.
        model_save_path: where to save the trained model.
        n_epochs: training epochs.
        episodes_per_round: RNG seeds per round for episode diversity.
        lr: learning rate.
        weight_decay: L2 regularization.
        val_rounds: rounds to hold out for validation. Default: [15, 16].
        device_name: "auto", "mps", "cuda", or "cpu".
        budget: query budget per episode.

    Returns:
        dict with training history.
    """
    # Device selection
    if device_name == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    print(f"Training on device: {device}")

    # Generate episodes
    print("Generating training episodes...", flush=True)
    gen = EpisodeGenerator(data_dir)
    all_rounds = sorted(gen.rounds.keys())

    if val_rounds is None:
        val_rounds = [r for r in all_rounds if r >= 15]
    train_rounds = [r for r in all_rounds if r not in val_rounds and r >= 6]

    print(f"  Train rounds: {train_rounds}")
    print(f"  Val rounds:   {val_rounds}", flush=True)

    t0 = time.time()
    train_episodes = gen.generate_batch(
        test_rounds=train_rounds,
        episodes_per_round=episodes_per_round,
        budget=budget,
    )
    val_episodes = gen.generate_batch(
        test_rounds=val_rounds,
        episodes_per_round=3,
        budget=budget,
        base_seed=9999,
    )
    t_gen = time.time() - t0
    print(f"  Generated {len(train_episodes)} train + {len(val_episodes)} val "
          f"episodes in {t_gen:.1f}s", flush=True)

    if not train_episodes:
        print("ERROR: No training episodes generated. Check data.")
        return {}

    # Convert to tensors
    train_data = episodes_to_tensors(train_episodes, device)
    val_data = episodes_to_tensors(val_episodes, device)

    # Model
    model_kwargs = dict(
        in_channels=27,
        base_channels=32,
        round_latent_dim=16,
        out_channels=6,
        residual_scale_init=0.01,
    )
    model = ResidualNeuralProcess(**model_kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr / 50)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_score": [], "v3_score": []}
    best_val = float("inf")
    patience_counter = 0
    patience = 60

    aug_rng = np.random.default_rng(42)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle episodes
        indices = np.random.permutation(len(train_data))

        for idx in indices:
            ep = train_data[idx]
            seeds = augment_episode(ep["seeds"], aug_rng)

            # Forward pass: process all seeds
            features_list = [s["features"] for s in seeds]
            v3_list = [s["v3_pred"] for s in seeds]

            preds = model(features_list, v3_list)

            # Loss: average across seeds
            loss = torch.tensor(0.0, device=device)
            for s_idx, pred in enumerate(preds):
                target = seeds[s_idx]["target"]
                loss = loss + entropy_weighted_kl_loss(pred, target)
            loss = loss / len(preds)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation
        if val_data:
            model.eval()
            val_loss = 0.0
            v3_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for ep in val_data:
                    seeds = ep["seeds"]
                    features_list = [s["features"] for s in seeds]
                    v3_list = [s["v3_pred"] for s in seeds]

                    preds = model(features_list, v3_list)

                    for s_idx, pred in enumerate(preds):
                        target = seeds[s_idx]["target"]
                        val_loss += entropy_weighted_kl_loss(pred, target).item()
                        v3_loss += entropy_weighted_kl_loss(
                            seeds[s_idx]["v3_pred"], target
                        ).item()
                        n_val += 1

            avg_val_loss = val_loss / max(n_val, 1)
            avg_v3_loss = v3_loss / max(n_val, 1)
            history["val_loss"].append(avg_val_loss)
            history["val_score"].append(competition_score_from_loss(avg_val_loss))
            history["v3_score"].append(competition_score_from_loss(avg_v3_loss))

            if avg_val_loss < best_val:
                best_val = avg_val_loss
                patience_counter = 0
                # Save best model
                Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "epoch": epoch,
                    "val_loss": best_val,
                    "val_score": history["val_score"][-1],
                    "v3_score": history["v3_score"][-1],
                    "train_rounds": train_rounds,
                    "val_rounds": val_rounds,
                }, model_save_path)
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:>4}/{n_epochs}: "
                      f"train={avg_train_loss:.4f}  "
                      f"val={avg_val_loss:.4f} (score≈{history['val_score'][-1]:.1f})  "
                      f"v3={avg_v3_loss:.4f} (score≈{history['v3_score'][-1]:.1f})  "
                      f"scale={model.residual_scale.item():.4f}", flush=True)

            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}", flush=True)
                break
        else:
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:>4}/{n_epochs}: train={avg_train_loss:.4f}",
                      flush=True)

    # Load best model
    if Path(model_save_path).exists():
        ckpt = torch.load(model_save_path, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"\nBest model at epoch {ckpt['epoch']+1}: "
              f"val_score≈{ckpt['val_score']:.1f} "
              f"(v3 baseline≈{ckpt['v3_score']:.1f})")
        print(f"  Δ ≈ {ckpt['val_score'] - ckpt['v3_score']:+.1f} points")
    else:
        # No validation data — save final model
        Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "model_kwargs": model_kwargs,
                    "epoch": n_epochs - 1}, model_save_path)

    print(f"Model saved to {model_save_path}")
    return history
