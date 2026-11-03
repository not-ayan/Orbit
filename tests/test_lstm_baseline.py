"""
Unit and integration tests for Sequence-to-Sequence LSTM Autoencoder and evaluation.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from orbit.models.lstm_autoencoder import LSTMAutoencoder
from orbit.training.trainer import ModelTrainer, set_seed
from orbit.evaluation.metrics import (
    select_validation_threshold,
    compute_event_metrics,
    evaluate_anomaly_predictions,
)
from orbit.evaluation.evaluator import AnomalyEvaluator


def test_lstm_autoencoder_forward_shape():
    B, L, F = 16, 128, 52
    model = LSTMAutoencoder(input_dim=F, hidden_dim=32, latent_dim=16, num_layers=1)
    x = torch.randn(B, L, F)
    x_hat = model(x)

    assert x_hat.shape == (B, L, F)
    latent = model.get_latent(x)
    assert latent.shape == (B, 16)


def test_reconstruction_error_reductions():
    B, L, F = 8, 64, 20
    model = LSTMAutoencoder(input_dim=F, hidden_dim=16, latent_dim=8, num_layers=1)
    x = torch.randn(B, L, F)

    err_none = model.compute_reconstruction_error(x, reduction="none")
    assert err_none.shape == (B, L, F)

    err_step = model.compute_reconstruction_error(x, reduction="feature_mean")
    assert err_step.shape == (B, L)

    err_win = model.compute_reconstruction_error(x, reduction="window_mean")
    assert err_win.shape == (B,)


def test_threshold_selection_and_event_metrics():
    # Synthetic normal validation scores
    val_scores = np.random.normal(loc=0.5, scale=0.1, size=1000)
    thresh_99 = select_validation_threshold(val_scores, method="percentile", percentile=99.0)
    assert thresh_99 > 0.5
    # Approximately 1% above threshold
    assert (val_scores >= thresh_99).mean() <= 0.02

    # Synthetic test ground truth with 2 events: [10:20] and [50:60]
    y_true = np.zeros(100, dtype=np.int64)
    y_true[10:20] = 1
    y_true[50:60] = 1

    # Prediction hits event 1 at index 12, and has a false alarm at index 80
    y_pred = np.zeros(100, dtype=np.int64)
    y_pred[12] = 1
    y_pred[80] = 1

    num_gt, detected, prec, rec, f1 = compute_event_metrics(y_true, y_pred)
    assert num_gt == 2
    assert detected == 1  # Detected first event, missed second
    assert rec == 0.5     # 1 / 2 detected
    assert prec == 0.5    # 1 out of 2 alarms hit a true event


def test_trainer_and_evaluator_integration(tmp_path):
    set_seed(42)
    B, L, F = 32, 32, 10
    N_train, N_val, N_test = 64, 32, 32

    # Create synthetic datasets
    X_train = torch.randn(N_train, L, F)
    y_train = torch.zeros(N_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=B)

    X_val = torch.randn(N_val, L, F)
    y_val = torch.zeros(N_val, dtype=torch.long)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=B)

    # Test set: inject large anomaly into half the samples
    X_test = torch.randn(N_test, L, F)
    y_test = torch.zeros(N_test, dtype=torch.long)
    X_test[16:] += 5.0
    y_test[16:] = 1
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=B)

    model = LSTMAutoencoder(input_dim=F, hidden_dim=16, latent_dim=8, num_layers=1)
    trainer = ModelTrainer(model, learning_rate=1e-2, patience=2, checkpoint_dir=tmp_path)

    # Train for 2 epochs
    history = trainer.fit(train_loader, val_loader, epochs=2)
    assert len(history["train_loss"]) == 2
    assert (tmp_path / "best_model.pt").exists()

    # Evaluate
    evaluator = AnomalyEvaluator(model)
    metrics = evaluator.evaluate(val_loader, test_loader, percentile=90.0, output_dir=tmp_path)

    assert metrics.auroc > 0.8  # Strong separation should be easily detected
    assert metrics.f1 > 0.5
    assert (tmp_path / "metrics.json").exists()

