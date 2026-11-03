"""
ORBIT: Self-Supervised Masked Reconstruction Pretraining for Industrial Telemetry.
Implements:
1. MaskGenerator: Generates contiguous block and token-level masking for multivariate time series.
2. MaskedTransformerAutoencoder: Self-supervised model computing loss strictly on masked positions.
3. Backbone extraction utilities for downstream cross-domain adaptation.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from orbit.models.temporal_encoder import (
    PositionalEncoding,
    DomainInputProjection,
    DomainReconstructionHead,
)


class MaskStrategy(str, Enum):
    BLOCK_TIME = "block_time"      # Contiguous spans of time steps
    RANDOM_TOKEN = "random_token"  # Independent random (time, sensor) positions
    HYBRID = "hybrid"              # Mix of contiguous spans and random sensors


class MaskGenerator:
    """
    Generates binary masks M in {0, 1}^(B, L, F) where M=1 denotes a masked position.
    Prevents trivial interpolation by supporting contiguous temporal block masking.
    """

    def __init__(
        self,
        strategy: Union[str, MaskStrategy] = MaskStrategy.HYBRID,
        mask_ratio: float = 0.20,
        min_block_len: int = 4,
        max_block_len: int = 16,
    ):
        self.strategy = MaskStrategy(strategy) if isinstance(strategy, str) else strategy
        self.mask_ratio = mask_ratio
        self.min_block_len = min_block_len
        self.max_block_len = max_block_len

    def generate_mask(
        self, x: torch.Tensor, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, L, F)
        Returns:
            mask: Binary tensor (B, L, F) with 1s at masked locations
        """
        B, L, F = x.shape
        device = device or x.device

        if self.strategy == MaskStrategy.RANDOM_TOKEN:
            # Random Bernoulli masking
            prob = torch.full((B, L, F), self.mask_ratio, device=device)
            mask = torch.bernoulli(prob)

        elif self.strategy == MaskStrategy.BLOCK_TIME:
            # Mask contiguous temporal blocks across all sensors
            mask = torch.zeros((B, L, F), device=device)
            target_masked_steps = int(L * self.mask_ratio)

            for b in range(B):
                masked_steps = 0
                while masked_steps < target_masked_steps:
                    block_len = np.random.randint(self.min_block_len, self.max_block_len + 1)
                    block_len = min(block_len, target_masked_steps - masked_steps, L)
                    start = np.random.randint(0, max(1, L - block_len + 1))
                    mask[b, start : start + block_len, :] = 1.0
                    masked_steps += block_len

        elif self.strategy == MaskStrategy.HYBRID:
            # Block masking on a random subset of sensors per window
            mask = torch.zeros((B, L, F), device=device)
            target_masked_steps = int(L * self.mask_ratio)

            for b in range(B):
                # Select 50-80% of sensors to receive temporal block masking
                num_sensors_to_mask = np.random.randint(int(F * 0.5), F + 1)
                sensor_indices = torch.randperm(F)[:num_sensors_to_mask]

                masked_steps = 0
                while masked_steps < target_masked_steps:
                    block_len = np.random.randint(self.min_block_len, self.max_block_len + 1)
                    block_len = min(block_len, target_masked_steps - masked_steps, L)
                    start = np.random.randint(0, max(1, L - block_len + 1))
                    mask[b, start : start + block_len, sensor_indices] = 1.0
                    masked_steps += block_len

                # Add light random masking (5%) to remaining positions
                extra_prob = torch.full((L, F), 0.05, device=device)
                extra_mask = torch.bernoulli(extra_prob)
                mask[b] = torch.clamp(mask[b] + extra_mask, 0.0, 1.0)

        # Ensure at least one token is masked per batch item to avoid zero denominator
        if mask.sum() == 0:
            mask[:, :self.min_block_len, :] = 1.0

        return mask


class ORBITPretrainer(nn.Module):
    """
    ORBIT Pretraining Model:
    Trains a temporal Transformer backbone using self-supervised masked telemetry reconstruction.
    Loss is calculated strictly on masked positions.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
        mask_strategy: Union[str, MaskStrategy] = MaskStrategy.HYBRID,
        mask_ratio: float = 0.20,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.mask_ratio = mask_ratio

        self.mask_generator = MaskGenerator(
            strategy=mask_strategy, mask_ratio=mask_ratio
        )

        # 1. Domain input projection (F -> d_model)
        self.input_projection = DomainInputProjection(
            input_dim=input_dim, d_model=d_model, dropout=dropout
        )

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=max_len, dropout=dropout
        )

        # 3. Backbone Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # 4. Reconstruction head (d_model -> F)
        self.recon_head = DomainReconstructionHead(
            d_model=d_model, output_dim=input_dim, dropout=dropout
        )

    def forward_pretrain(
        self, x: torch.Tensor, custom_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes masked forward pass and computes loss strictly on masked elements.

        Args:
            x: Raw input telemetry (B, L, F)
            custom_mask: Optional pre-defined mask (B, L, F)

        Returns:
            masked_loss: MSE computed ONLY on masked positions (scalar)
            x_hat: Full reconstructed sequence (B, L, F)
            mask: Binary mask applied (B, L, F)
        """
        # Generate mask
        if custom_mask is not None:
            mask = custom_mask
        else:
            mask = self.mask_generator.generate_mask(x)

        # Replace masked positions with zero (standardized telemetry has zero mean)
        x_corrupted = x * (1.0 - mask)

        # Forward through projection, positional encoding, backbone, and head
        h = self.input_projection(x_corrupted)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)
        x_hat = self.recon_head(encoded)

        # Compute MSE strictly on masked tokens
        sq_error = (x - x_hat) ** 2
        masked_error = sq_error * mask
        num_masked_tokens = mask.sum().clamp(min=1.0)
        masked_loss = masked_error.sum() / num_masked_tokens

        return masked_loss, x_hat, mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard unmasked forward pass (for inference / scoring)."""
        h = self.input_projection(x)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)
        x_hat = self.recon_head(encoded)
        return x_hat

    def get_backbone_state(self) -> Dict[str, torch.Tensor]:
        """
        Extracts the pretrained temporal backbone weights (Transformer encoder + positional encoding).
        These weights are domain-agnostic and ready for target adaptation.
        """
        return {
            "transformer_encoder": self.transformer_encoder.state_dict(),
            "pos_encoder": self.pos_encoder.state_dict(),
            "d_model": self.d_model,
            "num_layers": self.num_layers,
        }

    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """Inference reconstruction error."""
        x_hat = self.forward(x)
        sq_err = (x - x_hat) ** 2
        if reduction == "feature_mean":
            return sq_err.mean(dim=-1)
        elif reduction == "window_mean":
            return sq_err.mean(dim=[-2, -1])
        return sq_err


__all__ = [
    "MaskStrategy",
    "MaskGenerator",
    "ORBITPretrainer",
]

