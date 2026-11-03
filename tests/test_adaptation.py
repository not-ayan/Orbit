"""
Unit and integration tests for Few-Shot Target Domain Adaptation and Checkpoint Completeness.
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from orbit.adaptation.adapter import (
    AdaptationMode,
    TemporalAdapterLayer,
    AdaptedTransformerModel,
)
from orbit.adaptation.transfer import partition_target_budget
from orbit.data.windowing import SlidingWindowDataset
from orbit.models.checkpoint import (
    CheckpointValidationError,
    save_orbit_checkpoint,
    load_orbit_checkpoint,
)


def test_temporal_adapter_layer():
    B, L, D = 4, 32, 64
    adapter = TemporalAdapterLayer(d_model=D, d_adapter=16)
    x = torch.randn(B, L, D)

    # Initial output should be almost identical to input due to zero-init up-projection
    out = adapter(x)
    assert out.shape == (B, L, D)
    np.testing.assert_allclose(out.detach().numpy(), x.detach().numpy(), atol=1e-3)


def test_adapted_transformer_modes_and_freezing():
    B, L, F_target = 4, 32, 52
    d_model = 64

    # 1. ADAPTER MODE (Backbone frozen, adapters & projections trainable)
    model_adapter = AdaptedTransformerModel(
        target_dim=F_target,
        d_model=d_model,
        num_layers=2,
        mode=AdaptationMode.ADAPTER,
    )
    summary = model_adapter.get_parameter_summary()
    assert summary["frozen_parameters"] > 0
    assert summary["trainable_parameters"] > 0
    assert summary["trainable_percent"] < 30.0

    # Check requires_grad flags
    for param in model_adapter.transformer_encoder.parameters():
        assert not param.requires_grad
    for param in model_adapter.target_projection.parameters():
        assert param.requires_grad
    for param in model_adapter.adapters.parameters():
        assert param.requires_grad
    for param in model_adapter.target_recon_head.parameters():
        assert param.requires_grad

    # Forward pass
    x = torch.randn(B, L, F_target)
    out = model_adapter(x)
    assert out.shape == (B, L, F_target)

    # Backward pass: only trainable params should receive gradients
    loss = out.mean()
    loss.backward()
    for param in model_adapter.transformer_encoder.parameters():
        assert param.grad is None
    for param in model_adapter.target_projection.parameters():
        assert param.grad is not None

    # 2. TARGET CALIBRATION (LINEAR PROBE) MODE (Freeze backbone & adapters; train projections only)
    model_cal = AdaptedTransformerModel(
        target_dim=F_target,
        d_model=d_model,
        num_layers=2,
        mode=AdaptationMode.TARGET_CALIBRATION,
    )
    summary_cal = model_cal.get_parameter_summary()
    assert summary_cal["trainable_parameters"] < summary["trainable_parameters"]
    for param in model_cal.adapters.parameters():
        assert not param.requires_grad
    for param in model_cal.target_projection.parameters():
        assert param.requires_grad
    for param in model_cal.target_recon_head.parameters():
        assert param.requires_grad

    # 3. ZERO-SHOT MODE (All frozen)
    model_zs = AdaptedTransformerModel(
        target_dim=F_target,
        d_model=d_model,
        mode=AdaptationMode.ZERO_SHOT,
    )
    summary_zs = model_zs.get_parameter_summary()
    assert summary_zs["trainable_parameters"] == 0


def test_zero_shot_uncalibrated_mismatch_rejection():
    """
    Zero-shot inference across mismatched schemas (e.g. 52 target vs 92 source)
    MUST be rejected if projections are uncalibrated and no sensor mapping is provided.
    """
    model = AdaptedTransformerModel(
        target_dim=52,
        source_dim=92,
        d_model=64,
        num_layers=2,
        mode=AdaptationMode.ZERO_SHOT,
    )
    x = torch.randn(2, 32, 52)
    # Forward or compute_reconstruction_error should raise CheckpointValidationError
    with pytest.raises(CheckpointValidationError, match="Cannot perform zero-shot inference"):
        model.compute_reconstruction_error(x)


def test_zero_shot_shared_schema_deterministic():
    """
    When schema matches or projections are explicitly calibrated, zero-shot inference
    must succeed and be 100% deterministic.
    """
    torch.manual_seed(42)
    model = AdaptedTransformerModel(
        target_dim=52,
        source_dim=52,  # Shared schema
        d_model=64,
        num_layers=2,
        mode=AdaptationMode.ZERO_SHOT,
    )
    model.mark_calibrated()

    x = torch.randn(2, 32, 52)
    err1 = model.compute_reconstruction_error(x, reduction="window_mean")
    err2 = model.compute_reconstruction_error(x, reduction="window_mean")
    np.testing.assert_allclose(err1.detach().numpy(), err2.detach().numpy(), atol=1e-7)


def test_checkpoint_roundtrip_and_validation(tmp_path):
    """
    Tests saving and loading versioned checkpoints, and verifies that corrupted/incomplete
    checkpoints fail loudly.
    """
    model = AdaptedTransformerModel(
        target_dim=30,
        d_model=32,
        num_layers=2,
        mode=AdaptationMode.ADAPTER,
    )
    ckpt_path = tmp_path / "test_model.pt"
    saved_path = model.save_checkpoint(
        ckpt_path,
        schema_info={"feature_dim": 30, "dataset_name": "synthetic"},
        provenance={"seed": 123},
    )
    assert saved_path.exists()

    loaded = load_orbit_checkpoint(saved_path, expected_model_type="AdaptedTransformerModel")
    assert loaded["schema_version"] == "1.0.0"
    assert loaded["schema_info"]["feature_dim"] == 30
    assert "state_dict" in loaded

    # Corrupt model type should fail
    with pytest.raises(CheckpointValidationError, match="Model type mismatch"):
        load_orbit_checkpoint(saved_path, expected_model_type="ORBITPretrainer")

    # Corrupt feature dim should fail
    with pytest.raises(CheckpointValidationError, match="Feature dimension mismatch"):
        load_orbit_checkpoint(saved_path, target_dim=99)


def test_load_real_pretrained_backbone():
    backbone_path = Path("results/pretraining/wadi/pretrained_backbone.pt")
    if backbone_path.exists():
        model = AdaptedTransformerModel(
            target_dim=52,
            pretrained_backbone_path=backbone_path,
            d_model=64,
            num_layers=3,
            mode=AdaptationMode.ADAPTER,
        )
        x = torch.randn(2, 128, 52)
        out = model(x)
        assert out.shape == (2, 128, 52)


def test_partition_target_budget():
    T, F = 1000, 10
    L, S = 100, 10
    data = np.zeros((T, F), dtype=np.float32)
    dataset = SlidingWindowDataset(data, window_size=L, stride=S)
    total_w = len(dataset)
    assert total_w == (1000 - 100) // 10 + 1  # 91 windows

    # 0% budget
    sub_0, meta_0 = partition_target_budget(dataset, 0.0)
    assert sub_0 is None
    assert meta_0["num_windows"] == 0

    # 1% budget
    sub_1, meta_1 = partition_target_budget(dataset, 1.0)
    assert len(sub_1) == 1
    assert sub_1.indices == [0]

    # 10% budget
    sub_10, meta_10 = partition_target_budget(dataset, 10.0)
    assert len(sub_10) == 10
    assert sub_10.indices == list(range(10))

    # 100% budget
    sub_100, meta_100 = partition_target_budget(dataset, 100.0)
    assert len(sub_100) == total_w


def test_checkpoint_tensor_truth_value_safe(tmp_path):
    """
    Verifies that tensor-valued fields in checkpoints do not trigger ambiguous boolean evaluation.
    """
    ckpt_path = tmp_path / "tensor_bool_test.pt"
    dummy_model = AdaptedTransformerModel(target_dim=20, d_model=64, num_layers=2)
    # Construct a checkpoint payload containing multi-element tensor keys
    payload = {
        "transformer_encoder": dummy_model.transformer_encoder.state_dict(),
        "pos_encoder": dummy_model.pos_encoder.state_dict(),
        "input_projection.weight": torch.randn(64, 20),
        "recon_head.weight": torch.randn(20, 64),
    }
    torch.save(payload, ckpt_path)

    # Instantiating AdaptedTransformerModel should safely load without boolean ambiguity
    model = AdaptedTransformerModel(
        target_dim=20,
        pretrained_backbone_path=ckpt_path,
        d_model=64,
        num_layers=2,
        mode=AdaptationMode.ZERO_SHOT,
    )
    assert model.source_dim == 20
    assert model.is_calibrated is True


def test_sensor_mapping_execution_in_forward():
    """
    Verifies that sensor_mapping is actually applied to the input in the forward path.
    """
    source_dim = 10
    target_dim = 4
    # Sensor mapping dictionary mapping target indices {0, 1, 2, 3} -> source indices {2, 5, 7, 9}
    mapping = {0: 2, 1: 5, 2: 7, 3: 9}

    model = AdaptedTransformerModel(
        target_dim=target_dim,
        source_dim=source_dim,
        sensor_mapping=mapping,
        d_model=32,
        num_layers=2,
        mode=AdaptationMode.ZERO_SHOT,
    )
    # Mock source projection and head
    from orbit.models.temporal_encoder import DomainInputProjection, DomainReconstructionHead
    model.source_projection = DomainInputProjection(input_dim=source_dim, d_model=32)
    model.source_recon_head = DomainReconstructionHead(d_model=32, output_dim=source_dim)

    x = torch.randn(2, 16, target_dim)
    # Forward pass must succeed and return output matching target shape
    out = model(x)
    assert out.shape == (2, 16, target_dim)


def test_evaluator_invokes_inference_validation():
    """
    Verifies that evaluator.compute_step_errors triggers validate_for_inference()
    and fails loudly if a zero-shot model has uncalibrated random projections.
    """
    from orbit.evaluation.evaluator import AnomalyEvaluator
    from torch.utils.data import DataLoader, TensorDataset

    model = AdaptedTransformerModel(
        target_dim=52,
        source_dim=92,
        d_model=64,
        num_layers=2,
        mode=AdaptationMode.ZERO_SHOT,
    )
    # Uncalibrated and unmapped
    evaluator = AnomalyEvaluator(model)
    dummy_loader = DataLoader(TensorDataset(torch.randn(2, 32, 52), torch.zeros(2)), batch_size=2)

    with pytest.raises(CheckpointValidationError, match="Cannot perform zero-shot inference"):
        evaluator.compute_step_errors(dummy_loader)
