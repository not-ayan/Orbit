"""
Unit tests for configuration manager, dataset directory resolution, and checkpoint validation.
"""

from pathlib import Path
import pytest
import torch

from orbit.config import (
    ConfigValidationError,
    resolve_dataset_dir,
    validate_checkpoint_for_transfer,
    load_yaml_config,
)
from orbit.models.checkpoint import save_orbit_checkpoint
from orbit.adaptation.adapter import AdaptedTransformerModel, AdaptationMode


def test_resolve_dataset_dir_portable_custom(tmp_path):
    mock_data = tmp_path / "mock_swat_dir"
    mock_data.mkdir()
    resolved = resolve_dataset_dir("swat", custom_dir=mock_data)
    assert resolved == mock_data
    assert resolved.exists()


def test_resolve_dataset_dir_known_if_present():
    try:
        swat_dir = resolve_dataset_dir("swat")
        assert swat_dir.exists()
        assert "SWaT" in str(swat_dir)
    except ConfigValidationError:
        pytest.skip("SWaT raw dataset not available locally (portable test).")


def test_resolve_dataset_dir_unknown_raises():
    with pytest.raises(ConfigValidationError, match="Unknown dataset 'unknown_plant'"):
        resolve_dataset_dir("unknown_plant")


def test_resolve_dataset_dir_missing_custom_raises():
    with pytest.raises(ConfigValidationError, match="Dataset directory for 'custom' not found"):
        resolve_dataset_dir("custom", custom_dir="non_existent/path/to/telemetry")


def test_validate_checkpoint_for_transfer_success(tmp_path):
    model = AdaptedTransformerModel(target_dim=10, d_model=32, num_layers=2)
    ckpt_path = tmp_path / "test_backbone.pt"
    save_orbit_checkpoint(
        model=model,
        filepath=ckpt_path,
        architecture={"d_model": 32, "num_layers": 2},
        schema_info={"dataset_name": "wadi", "feature_dim": 10},
    )

    # Validation should succeed
    data = validate_checkpoint_for_transfer(
        ckpt_path,
        expected_source_dataset="wadi",
        expected_d_model=32,
    )
    assert data["architecture"]["d_model"] == 32


def test_validate_checkpoint_for_transfer_mismatches(tmp_path):
    model = AdaptedTransformerModel(target_dim=10, d_model=32, num_layers=2)
    ckpt_path = tmp_path / "test_backbone.pt"
    save_orbit_checkpoint(
        model=model,
        filepath=ckpt_path,
        architecture={"d_model": 32, "num_layers": 2},
        schema_info={"dataset_name": "wadi", "feature_dim": 10},
    )

    # Dataset mismatch should fail loudly
    with pytest.raises(ConfigValidationError, match="Checkpoint dataset mismatch"):
        validate_checkpoint_for_transfer(
            ckpt_path,
            expected_source_dataset="swat",  # Mismatch!
        )

    # d_model mismatch should fail loudly
    with pytest.raises(ConfigValidationError, match="Architecture mismatch"):
        validate_checkpoint_for_transfer(
            ckpt_path,
            expected_d_model=128,  # Mismatch!
        )


def test_load_yaml_config():
    cfg = load_yaml_config("configs/transfer_config.yaml")
    assert "experiment" in cfg
    assert cfg["experiment"]["source_dataset"] == "wadi"
    assert cfg["experiment"]["target_dataset"] == "swat"
    assert "seeds" in cfg["experiment"]

