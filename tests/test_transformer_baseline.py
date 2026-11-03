"""
Unit and integration tests for Multivariate Temporal Transformer Autoencoder.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from orbit.models.temporal_encoder import (
    PositionalEncoding,
    DomainInputProjection,
    DomainReconstructionHead,
    TransformerAutoencoder,
)
from orbit.training.trainer import ModelTrainer, set_seed
from orbit.evaluation.evaluator import AnomalyEvaluator


def test_domain_projections():
    B, L, F = 16, 128, 52
    d_model = 64

    proj = DomainInputProjection(input_dim=F, d_model=d_model)
    x = torch.randn(B, L, F)
    h = proj(x)
    assert h.shape == (B, L, d_model)

    head = DomainReconstructionHead(d_model=d_model, output_dim=F)
    x_hat = head(h)
    assert x_hat.shape == (B, L, F)


def test_positional_encoding():
    B, L, d_model = 8, 128, 64
    pe = PositionalEncoding(d_model=d_model, max_len=256, dropout=0.0)
    x = torch.zeros(B, L, d_model)
    x_pe = pe(x)

    assert x_pe.shape == (B, L, d_model)
    # Consecutive positions should have distinct encodings
    assert not torch.allclose(x_pe[:, 0, :], x_pe[:, 1, :])


def test_transformer_autoencoder_forward():
    B, L, F = 8, 128, 92
    model = TransformerAutoencoder(
        input_dim=F,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    x = torch.randn(B, L, F)
    x_hat = model(x)
    assert x_hat.shape == (B, L, F)

    # Latent encoding shape
    encoded = model.encode(x)
    assert encoded.shape == (B, L, 64)

    # Reductions
    err_step = model.compute_reconstruction_error(x, reduction="feature_mean")
    assert err_step.shape == (B, L)
    err_win = model.compute_reconstruction_error(x, reduction="window_mean")
    assert err_win.shape == (B,)


def test_transformer_trainer_and_evaluator_integration(tmp_path):
    set_seed(42)
    B, L, F = 16, 32, 10
    N_train, N_val, N_test = 64, 32, 32

    # Synthetic datasets
    X_train = torch.randn(N_train, L, F)
    y_train = torch.zeros(N_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=B)

    X_val = torch.randn(N_val, L, F)
    y_val = torch.zeros(N_val, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=B)

    # Test set: inject anomaly burst into second half
    X_test = torch.randn(N_test, L, F)
    y_test = torch.zeros(N_test, dtype=torch.long)
    X_test[16:] += 4.0
    y_test[16:] = 1
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=B)

    model = TransformerAutoencoder(
        input_dim=F,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        dropout=0.0,
    )
    trainer = ModelTrainer(model, learning_rate=1e-3, patience=2, checkpoint_dir=tmp_path)

    # Train for 2 epochs
    history = trainer.fit(train_loader, val_loader, epochs=2)
    assert len(history["train_loss"]) == 2
    assert (tmp_path / "best_model.pt").exists()

    # Evaluate
    evaluator = AnomalyEvaluator(model)
    metrics = evaluator.evaluate(val_loader, test_loader, percentile=90.0, output_dir=tmp_path)

    assert metrics.auroc > 0.75
    assert (tmp_path / "metrics.json").exists()

