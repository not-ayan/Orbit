"""
Unit and integration tests for Self-Supervised Masked Reconstruction Pretraining.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from orbit.models.orbit import (
    MaskStrategy,
    MaskGenerator,
    ORBITPretrainer,
)
from orbit.training.pretrain_trainer import MaskedPretrainer
from orbit.training.trainer import set_seed


def test_mask_generator_strategies():
    B, L, F = 8, 128, 52
    x = torch.zeros(B, L, F)

    # Random token
    gen_random = MaskGenerator(strategy=MaskStrategy.RANDOM_TOKEN, mask_ratio=0.2)
    m_rand = gen_random.generate_mask(x)
    assert m_rand.shape == (B, L, F)
    assert ((m_rand == 0) | (m_rand == 1)).all()
    # Mask ratio should be approximately 0.20
    assert 0.15 <= m_rand.float().mean().item() <= 0.25

    # Block time
    gen_block = MaskGenerator(strategy=MaskStrategy.BLOCK_TIME, mask_ratio=0.25)
    m_block = gen_block.generate_mask(x)
    assert m_block.shape == (B, L, F)
    assert m_block.sum() > 0

    # Hybrid
    gen_hybrid = MaskGenerator(strategy=MaskStrategy.HYBRID, mask_ratio=0.20)
    m_hyb = gen_hybrid.generate_mask(x)
    assert m_hyb.shape == (B, L, F)
    assert m_hyb.sum() > 0


def test_orbit_pretrainer_masked_loss_computation():
    B, L, F = 4, 32, 10
    model = ORBITPretrainer(
        input_dim=F,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        mask_ratio=0.25,
    )

    x = torch.randn(B, L, F)
    # Custom fixed mask where only the first 4 timesteps of the first batch item are masked
    custom_mask = torch.zeros(B, L, F)
    custom_mask[0, :4, :] = 1.0

    loss, x_hat, mask = model.forward_pretrain(x, custom_mask=custom_mask)

    assert loss.dim() == 0  # Scalar
    assert not torch.isnan(loss)
    assert loss.item() > 0.0
    assert x_hat.shape == (B, L, F)

    # Manual MSE verification on the masked region only
    expected_sq_err = ((x[0, :4, :] - x_hat[0, :4, :]) ** 2).mean()
    np.testing.assert_allclose(loss.item(), expected_sq_err.item(), rtol=1e-4)


def test_orbit_backbone_export():
    B, L, F = 2, 16, 8
    model = ORBITPretrainer(input_dim=F, d_model=32, nhead=2, num_layers=2)

    backbone_state = model.get_backbone_state()
    assert "transformer_encoder" in backbone_state
    assert "pos_encoder" in backbone_state
    assert backbone_state["d_model"] == 32
    assert backbone_state["num_layers"] == 2


def test_masked_pretrainer_integration(tmp_path):
    set_seed(42)
    B, L, F = 16, 32, 10
    N_train, N_val = 64, 32

    X_train = torch.randn(N_train, L, F)
    train_loader = DataLoader(TensorDataset(X_train, torch.zeros(N_train)), batch_size=B)
    X_val = torch.randn(N_val, L, F)
    val_loader = DataLoader(TensorDataset(X_val, torch.zeros(N_val)), batch_size=B)

    model = ORBITPretrainer(
        input_dim=F,
        d_model=32,
        nhead=2,
        num_layers=1,
        dim_feedforward=64,
        mask_ratio=0.20,
    )
    trainer = MaskedPretrainer(
        model=model,
        learning_rate=1e-3,
        patience=2,
        checkpoint_dir=tmp_path,
    )

    history = trainer.fit(train_loader, val_loader, epochs=2)
    assert len(history["masked_train_loss"]) == 2
    assert (tmp_path / "best_pretrain_model.pt").exists()
    assert (tmp_path / "pretrained_backbone.pt").exists()
    assert (tmp_path / "pretrain_history.json").exists()

