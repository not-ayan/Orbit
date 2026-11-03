"""
Multivariate Temporal Transformer Encoder for Industrial Telemetry.
Architecture:
1. DomainInputProjection: Projects domain-specific sensor dimension F to shared latent d_model.
2. PositionalEncoding: Injects temporal sequence ordering into representations.
3. TemporalTransformerEncoder: Multi-layer bidirectional self-attention across time.
4. DomainReconstructionHead: Projects latent states back to original sensor dimension F.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for temporal sequences.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class DomainInputProjection(nn.Module):
    """
    Domain-specific input projection adapting variable sensor dimensions F
    into the unified Transformer latent dimension d_model.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        if hidden_dim is not None:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, L, F)
        Returns:
            Tensor of shape (B, L, d_model)
        """
        return self.proj(x)


class DomainReconstructionHead(nn.Module):
    """
    Domain-specific reconstruction head mapping latent states back to sensor dimension F.
    """

    def __init__(
        self,
        d_model: int = 64,
        output_dim: int = 52,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        if hidden_dim is not None:
            self.head = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.head = nn.Linear(d_model, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Latent tensor of shape (B, L, d_model)
        Returns:
            x_hat: Reconstructed telemetry of shape (B, L, F)
        """
        return self.head(h)


class TransformerAutoencoder(nn.Module):
    """
    Multivariate Temporal Transformer Autoencoder.
    Trained from scratch on normal telemetry to model normal temporal dependencies.
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
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers

        # 1. Domain-specific projection
        self.input_projection = DomainInputProjection(
            input_dim=input_dim, d_model=d_model, dropout=dropout
        )

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model, max_len=max_len, dropout=dropout
        )

        # 3. Stacked Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False,
        )

        # 4. Reconstruction head
        self.recon_head = DomainReconstructionHead(
            d_model=d_model, output_dim=input_dim, dropout=dropout
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input telemetry of shape (B, L, F)
            mask: Optional key padding mask or attention mask
        Returns:
            x_hat: Reconstructed sequence of shape (B, L, F)
        """
        # Project F -> d_model: (B, L, d_model)
        h = self.input_projection(x)
        # Add positional encoding
        h = self.pos_encoder(h)
        # Apply temporal self-attention
        encoded = self.transformer_encoder(h, mask=mask)
        # Reconstruct back to sensor dimension: (B, L, F)
        x_hat = self.recon_head(encoded)
        return x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent temporal representations (B, L, d_model)."""
        h = self.input_projection(x)
        h = self.pos_encoder(h)
        return self.transformer_encoder(h)

    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """
        Computes squared reconstruction error (x - x_hat)^2.
        """
        x_hat = self.forward(x)
        sq_err = (x - x_hat) ** 2
        if reduction == "feature_mean":
            return sq_err.mean(dim=-1)
        elif reduction == "window_mean":
            return sq_err.mean(dim=[-2, -1])
        return sq_err


__all__ = [
    "PositionalEncoding",
    "DomainInputProjection",
    "DomainReconstructionHead",
    "TransformerAutoencoder",
]
