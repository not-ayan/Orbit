"""
Sequence-to-Sequence LSTM Autoencoder Baseline for Multivariate Telemetry.
Architecture:
Encoder: Stacked LSTM encoding sequence (B, L, F) into a bottleneck latent vector (B, latent_dim).
Decoder: Unfolds latent representation across sequence length L and reconstructs (B, L, F).
"""

from typing import Tuple, Dict, Any
import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    Multivariate sequence encoder using stacked LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (B, L, F)
        Returns:
            latent: Bottleneck representation (B, latent_dim)
            hidden: Final LSTM states (h_n, c_n)
        """
        # lstm_out: (B, L, hidden_dim), (h_n, c_n): (num_layers, B, hidden_dim)
        _, (h_n, c_n) = self.lstm(x)
        # Use last layer hidden state: (B, hidden_dim)
        last_hidden = h_n[-1]
        latent = self.fc_latent(last_hidden)
        return latent, (h_n, c_n)


class LSTMDecoder(nn.Module):
    """
    Multivariate sequence decoder reconstructing telemetry from latent vector.
    """

    def __init__(
        self,
        output_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.fc_in = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, latent: torch.Tensor, target_len: int) -> torch.Tensor:
        """
        Args:
            latent: Bottleneck tensor (B, latent_dim)
            target_len: Desired sequence length L
        Returns:
            x_recon: Reconstructed sequence (B, L, F)
        """
        # Repeat latent vector across sequence length L: (B, L, latent_dim)
        repeated_latent = latent.unsqueeze(1).repeat(1, target_len, 1)

        # Decode sequence through LSTM
        dec_out, _ = self.lstm(repeated_latent)  # (B, L, hidden_dim)
        x_recon = self.fc_out(dec_out)           # (B, L, output_dim)
        return x_recon


class LSTMAutoencoder(nn.Module):
    """
    Complete Sequence-to-Sequence LSTM Autoencoder model.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = LSTMDecoder(
            output_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass reconstructs the input sequence.
        Args:
            x: Input tensor (B, L, F)
        Returns:
            x_hat: Reconstructed tensor (B, L, F)
        """
        target_len = x.size(1)
        latent, _ = self.encoder(x)
        x_hat = self.decoder(latent, target_len=target_len)
        return x_hat

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck representation."""
        latent, _ = self.encoder(x)
        return latent

    def compute_reconstruction_error(
        self, x: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """
        Computes squared reconstruction error (x - x_hat)^2.
        Args:
            x: Input tensor (B, L, F)
            reduction:
                'none' -> (B, L, F)
                'feature_mean' -> (B, L) error per timestep across sensors
                'window_mean' -> (B,) mean error across entire window
        """
        x_hat = self.forward(x)
        sq_err = (x - x_hat) ** 2
        if reduction == "feature_mean":
            return sq_err.mean(dim=-1)
        elif reduction == "window_mean":
            return sq_err.mean(dim=[-2, -1])
        return sq_err


__all__ = [
    "LSTMEncoder",
    "LSTMDecoder",
    "LSTMAutoencoder",
]

