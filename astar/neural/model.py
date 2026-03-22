"""
Residual Neural Process for Astar Island.

Architecture:
  - U-Net-style CNN backbone over per-cell features (27 channels)
  - Round-latent vector tiled into bottleneck (cross-seed information)
  - Residual head: predicts Δlogits (6 channels) to correct v3 posterior
  - Final prediction: softmax(log(v3_posterior) + scale * Δlogits)

This is a *residual* model: it learns corrections to the v3 Dirichlet
baseline rather than predicting from scratch. This is critical because
we only have ~16 rounds of training data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def stable_softmax(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Manual softmax that works correctly on MPS (F.softmax has MPS bugs)."""
    x_shifted = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


class ConvBlock(nn.Module):
    """Conv → BatchNorm → GELU → Conv → BatchNorm → GELU + residual + dropout."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.dropout(h)
        h = self.bn2(self.conv2(h))
        return F.gelu(h + self.skip(x))


class SeedPooler(nn.Module):
    """Pool features across seeds to infer a shared round latent.

    Uses mean + max pooling over seed embeddings, then projects to
    a round-latent vector that gets tiled back to spatial dimensions.
    """

    def __init__(self, spatial_ch: int, round_latent_dim: int = 32):
        super().__init__()
        # Global average pool each seed's feature map → (B*S, spatial_ch)
        # Then pool across S seeds
        self.proj = nn.Sequential(
            nn.Linear(spatial_ch * 2, round_latent_dim),  # mean + max
            nn.GELU(),
            nn.Linear(round_latent_dim, round_latent_dim),
        )

    def forward(self, seed_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            seed_features: list of (B, C, H, W) tensors, one per seed.

        Returns:
            (B, round_latent_dim) round-level context.
        """
        # Global average pool each seed
        pooled = []
        for feat in seed_features:
            mean_pool = feat.mean(dim=(2, 3))   # (B, C)
            max_pool = feat.amax(dim=(2, 3))    # (B, C)
            pooled.append(torch.cat([mean_pool, max_pool], dim=-1))  # (B, 2C)

        # Stack and pool across seeds: (B, S, 2C)
        stacked = torch.stack(pooled, dim=1)
        # Mean across seeds
        seed_mean = stacked.mean(dim=1)  # (B, 2C)
        return self.proj(seed_mean)  # (B, round_latent_dim)


class ResidualNeuralProcess(nn.Module):
    """Residual model on top of v3 Dirichlet baseline.

    Input per cell (27 channels):
      [0:8]   terrain one-hot
      [8:16]  observation counts (8-state)
      [16]    total observation count
      [17:23] v3 posterior 6-class
      [23]    v3 entropy
      [24]    coastal flag
      [25]    distance to settlement (norm)
      [26]    settlement density (norm)

    Output per cell (6 channels):
      Δlogits to add to log(v3_posterior)

    Architecture: small U-Net with round-latent injection.
    """

    def __init__(
        self,
        in_channels: int = 27,
        base_channels: int = 48,
        round_latent_dim: int = 32,
        out_channels: int = 6,
        residual_scale_init: float = 0.1,
    ):
        super().__init__()

        ch = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, ch)
        self.enc2 = ConvBlock(ch, ch * 2)
        self.enc3 = ConvBlock(ch * 2, ch * 4)

        # Bottleneck (with round latent injection)
        self.bottleneck = ConvBlock(ch * 4 + round_latent_dim, ch * 4)

        # Decoder
        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.dec3 = ConvBlock(ch * 4, ch * 2)  # skip connection doubles channels

        self.up2 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.dec2 = ConvBlock(ch * 2, ch)  # skip connection

        # Final head: predict residual logits
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, out_channels, 1),
        )
        # Zero-init output layer so initial Δlogits ≈ 0 (preserves v3 baseline)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # Seed pooler for round latent
        self.seed_pooler = SeedPooler(ch * 4, round_latent_dim)

        # Fixed residual scale (small to not destroy v3 baseline at start)
        self.residual_scale = nn.Parameter(
            torch.tensor(residual_scale_init)
        )

        self.pool = nn.MaxPool2d(2, 2)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder, return (bottleneck_input, skip2, skip1)."""
        e1 = self.enc1(x)          # (B, ch, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 2ch, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 4ch, H/4, W/4)
        return e3, e2, e1

    def decode(
        self,
        bottleneck: torch.Tensor,
        skip2: torch.Tensor,
        skip1: torch.Tensor,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Run decoder with skip connections."""
        # Up + skip
        d3 = self.up3(bottleneck)
        # Handle size mismatches from non-power-of-2 dimensions
        d3 = F.interpolate(d3, size=skip2.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, skip2], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=skip1.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, skip1], dim=1))

        return self.head(d2)  # (B, 6, H, W)

    def forward_single_seed(
        self,
        features: torch.Tensor,
        round_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for one seed.

        Args:
            features: (B, 27, H, W) per-cell feature tensor.
            round_latent: (B, round_latent_dim) from seed pooler.

        Returns:
            (B, 6, H, W) residual logits.
        """
        B = features.shape[0]
        H, W = features.shape[2], features.shape[3]

        e3, e2, e1 = self.encode(features)

        # Inject round latent into bottleneck
        rl = round_latent[:, :, None, None].expand(-1, -1, e3.shape[2], e3.shape[3])
        bottleneck_in = torch.cat([e3, rl], dim=1)
        bottleneck = self.bottleneck(bottleneck_in)

        delta_logits = self.decode(bottleneck, e2, e1, H, W)
        return torch.clamp(delta_logits * self.residual_scale, -3.0, 3.0)

    def forward(
        self,
        features_list: list[torch.Tensor],
        v3_preds_list: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Forward pass for all seeds in a round.

        Args:
            features_list: list of (B, 27, H, W) per seed.
            v3_preds_list: list of (B, 6, H, W) v3 posterior probs per seed.

        Returns:
            list of (B, 6, H, W) corrected probability tensors per seed.
        """
        # Step 1: encode all seeds
        encoded = [self.encode(f) for f in features_list]
        bottleneck_features = [e[0] for e in encoded]

        # Step 2: pool across seeds for round latent
        round_latent = self.seed_pooler(bottleneck_features)

        # Step 3: decode each seed with shared round latent
        predictions = []
        for i, (features, v3_pred) in enumerate(zip(features_list, v3_preds_list)):
            e3, e2, e1 = encoded[i]
            H, W = features.shape[2], features.shape[3]

            # Inject round latent
            rl = round_latent[:, :, None, None].expand(-1, -1, e3.shape[2], e3.shape[3])
            bottleneck_in = torch.cat([e3, rl], dim=1)
            bottleneck = self.bottleneck(bottleneck_in)

            delta_logits = self.decode(bottleneck, e2, e1, H, W)
            delta_logits = torch.clamp(delta_logits * self.residual_scale, -3.0, 3.0)

            # Apply residual: softmax(log(v3) + delta)
            v3_safe = torch.clamp(v3_pred, min=1e-8)
            logits = torch.log(v3_safe) + delta_logits
            pred = stable_softmax(logits, dim=1)  # (B, 6, H, W)
            predictions.append(pred)

        return predictions

    def predict_corrected(
        self,
        features: torch.Tensor,
        v3_pred: torch.Tensor,
        round_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Predict for a single seed with pre-computed round latent.

        Args:
            features: (1, 27, H, W)
            v3_pred: (1, 6, H, W) v3 posterior probabilities
            round_latent: (1, round_latent_dim)

        Returns:
            (1, 6, H, W) corrected probabilities
        """
        delta = self.forward_single_seed(features, round_latent)
        v3_safe = torch.clamp(v3_pred, min=1e-8)
        logits = torch.log(v3_safe) + delta
        return stable_softmax(logits, dim=1)
