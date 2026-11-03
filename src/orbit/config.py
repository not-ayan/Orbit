"""
Centralized configuration manager, dataset directory resolver, and checkpoint validator for ORBIT.
Removes hardcoded assumptions and enforces strict source/target/checkpoint compatibility.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import torch

from orbit.models.checkpoint import (
    CheckpointValidationError,
    load_orbit_checkpoint,
)


class ConfigValidationError(ValueError):
    """Raised when an experiment configuration or dataset path is invalid or missing."""
    pass


# Canonical default dataset registry (relative to project root)
KNOWN_DATASETS = {
    "swat": Path("datasets/SWaT.A8_June 2021/For Mark"),
    "swat_a8": Path("datasets/SWaT.A8_June 2021/For Mark"),
    "wadi": Path("datasets/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019"),
    "wadi_a2": Path("datasets/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019"),
}


def resolve_dataset_dir(
    dataset_name: str,
    custom_dir: Optional[Union[str, Path]] = None,
    project_root: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolves and validates the directory path for a dataset.

    Args:
        dataset_name: Dataset identifier (e.g. 'swat', 'wadi').
        custom_dir: Optional explicit user-provided directory path.
        project_root: Project root folder (defaults to current working directory).

    Returns:
        Validated Path to the dataset directory.
    """
    root = Path(project_root) if project_root else Path.cwd()

    if custom_dir:
        path = Path(custom_dir)
        if not path.is_absolute():
            path = root / path
    else:
        name_lower = dataset_name.lower().strip()
        if name_lower not in KNOWN_DATASETS:
            raise ConfigValidationError(
                f"Unknown dataset '{dataset_name}'. Must be one of {list(KNOWN_DATASETS.keys())} "
                f"or specified explicitly via --data-dir."
            )
        path = root / KNOWN_DATASETS[name_lower]

    if not path.exists():
        raise ConfigValidationError(
            f"Dataset directory for '{dataset_name}' not found at: {path}. "
            f"Please verify the path or supply --data-dir."
        )

    return path


def validate_checkpoint_for_transfer(
    checkpoint_path: Union[str, Path],
    expected_source_dataset: Optional[str] = None,
    expected_d_model: Optional[int] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """
    Validates a pretrained checkpoint before using it in a transfer experiment.
    Fails loudly with actionable error messages if there is a domain or architecture mismatch.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise ConfigValidationError(
            f"Pretrained checkpoint not found at: {path}. "
            f"Run pretraining first or provide a valid --backbone-path."
        )

    ckpt_data = load_orbit_checkpoint(path, map_location=map_location, strict=False)

    # Validate source dataset if recorded in checkpoint metadata
    schema_info = ckpt_data.get("schema_info", {})
    recorded_dataset = schema_info.get("dataset_name")
    if recorded_dataset and expected_source_dataset:
        if recorded_dataset.lower() != expected_source_dataset.lower():
            raise ConfigValidationError(
                f"Checkpoint dataset mismatch: checkpoint was trained on '{recorded_dataset}', "
                f"but experiment specified source dataset '{expected_source_dataset}'."
            )

    # Validate architecture d_model if recorded
    arch = ckpt_data.get("architecture", {})
    recorded_d_model = arch.get("d_model")
    if recorded_d_model and expected_d_model and recorded_d_model != expected_d_model:
        raise ConfigValidationError(
            f"Architecture mismatch: checkpoint d_model={recorded_d_model}, "
            f"but experiment configured d_model={expected_d_model}."
        )

    return ckpt_data


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads a YAML experiment configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigValidationError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


__all__ = [
    "ConfigValidationError",
    "KNOWN_DATASETS",
    "resolve_dataset_dir",
    "validate_checkpoint_for_transfer",
    "load_yaml_config",
]

