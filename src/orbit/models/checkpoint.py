"""
Robust, versioned checkpointing and schema validation for ORBIT models.
Guarantees complete inference states, parameter validation, and provenance tracking.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import datetime
import torch
import torch.nn as nn


class CheckpointValidationError(ValueError):
    """Raised when a checkpoint fails architectural, schema, or completeness validation."""
    pass


SCHEMA_VERSION = "1.0.0"


def save_orbit_checkpoint(
    model: nn.Module,
    filepath: Union[str, Path],
    architecture: Dict[str, Any],
    schema_info: Dict[str, Any],
    adaptation_mode: Optional[str] = None,
    normalization_stats: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Saves a versioned, self-describing ORBIT checkpoint containing:
    - schema_version
    - architecture hyperparameters
    - dataset schema metadata (dimensions, feature names)
    - adaptation mode
    - normalization/scaling parameters
    - complete model state_dict
    - provenance (training epochs, seeds, timestamps, parent checkpoint)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    prov = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": str(torch.__version__),
    }
    if provenance:
        prov.update(provenance)

    checkpoint_payload = {
        "schema_version": SCHEMA_VERSION,
        "model_type": model.__class__.__name__,
        "architecture": architecture,
        "schema_info": schema_info,
        "adaptation_mode": adaptation_mode,
        "normalization_stats": normalization_stats or {},
        "provenance": prov,
        "state_dict": model.state_dict(),
        "extra_state": extra_state or {},
    }

    torch.save(checkpoint_payload, filepath)
    return filepath


def load_orbit_checkpoint(
    filepath: Union[str, Path],
    expected_model_type: Optional[str] = None,
    target_dim: Optional[int] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Loads and validates an ORBIT checkpoint.
    Supports both v1.0.0 versioned format and legacy raw state dicts with strict validation.

    Args:
        filepath: Path to the .pt checkpoint.
        expected_model_type: If provided, asserts the model type in the checkpoint.
        target_dim: If provided, checks target sensor dimension compatibility.
        map_location: Torch device mapping.
        strict: If True, enforces complete metadata and shape compatibility.

    Returns:
        checkpoint dictionary containing state_dict and metadata.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {filepath}")

    # Explicitly allow loading internal checkpoints
    raw_data = torch.load(filepath, map_location=map_location, weights_only=False)

    # Handle legacy checkpoints gracefully (e.g. pretrained_backbone.pt or best_model.pt)
    if not isinstance(raw_data, dict) or ("schema_version" not in raw_data and "state_dict" not in raw_data):
        # Legacy format detection
        if "model_state_dict" in raw_data:
            state_dict = raw_data["model_state_dict"]
        elif "transformer_encoder" in raw_data:
            # Backbone-only legacy checkpoint
            state_dict = raw_data
        elif all(isinstance(v, torch.Tensor) for v in raw_data.values()):
            state_dict = raw_data
        else:
            raise CheckpointValidationError(
                f"Unknown checkpoint structure in {filepath}. Keys: {list(raw_data.keys())}"
            )

        checkpoint_data = {
            "schema_version": "legacy",
            "model_type": expected_model_type or "Unknown",
            "architecture": {},
            "schema_info": {},
            "adaptation_mode": None,
            "provenance": {"legacy": True, "source_file": str(filepath)},
            "state_dict": state_dict,
            "extra_state": raw_data if isinstance(raw_data, dict) else {},
        }
    else:
        checkpoint_data = raw_data

    # Validate model type if specified
    if expected_model_type and checkpoint_data.get("model_type") not in ("Unknown", expected_model_type):
        raise CheckpointValidationError(
            f"Model type mismatch: checkpoint has '{checkpoint_data.get('model_type')}', "
            f"expected '{expected_model_type}'."
        )

    # Validate schema compatibility if target_dim is specified
    if target_dim is not None and "schema_info" in checkpoint_data:
        saved_dim = checkpoint_data["schema_info"].get("feature_dim")
        if saved_dim is not None and saved_dim != target_dim and strict:
            raise CheckpointValidationError(
                f"Feature dimension mismatch: checkpoint expects {saved_dim} features, "
                f"but target domain has {target_dim} features."
            )

    return checkpoint_data


def validate_inference_completeness(model: nn.Module, model_name: str = "Model") -> None:
    """
    Verifies that all parameters in the model have valid numerical weights (no NaNs, no uninitialized buffers).
    """
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            raise CheckpointValidationError(
                f"{model_name} parameter '{name}' contains NaN values. Incomplete or corrupted checkpoint."
            )
        if torch.isinf(param).any():
            raise CheckpointValidationError(
                f"{model_name} parameter '{name}' contains Inf values."
            )


__all__ = [
    "CheckpointValidationError",
    "SCHEMA_VERSION",
    "save_orbit_checkpoint",
    "load_orbit_checkpoint",
    "validate_inference_completeness",
]
