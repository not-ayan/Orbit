"""
Unified, zero-leakage industrial time-series data pipeline.
Integrates loading, chronological splitting, train-only scaling, windowing, and artifact persistence.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from orbit.data.schema import WADI_A2_SCHEMA, SWAT_A8_SCHEMA
from orbit.data.loading import load_raw_wadi, load_raw_swat
from orbit.data.preprocessing import (
    split_chronological_normal,
    handle_missing_values,
    detect_constant_columns,
    IndustrialScaler,
)
from orbit.data.windowing import SlidingWindowDataset, LabelPolicy


@dataclass
class PipelineArtifacts:
    dataset_name: str
    sensor_columns: List[str]
    dropped_constant_columns: List[str]
    num_train_timesteps: int
    num_val_timesteps: int
    num_test_timesteps: int
    num_train_windows: int
    num_val_windows: int
    num_test_windows: int
    test_anomaly_ratio: float
    window_size: int
    stride_train: int
    stride_test: int
    label_policy: str
    scaler_path: str


class TelemetryPipeline:
    """
    End-to-end data pipeline managing preprocessing, zero-leakage splits,
    and PyTorch DataLoader generation.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: Union[str, Path],
        window_size: int = 128,
        stride_train: int = 32,
        stride_test: int = 32,
        val_ratio: float = 0.2,
        scale_method: str = "standard",
        drop_constants: bool = True,
        label_policy: str = "any",
        batch_size: int = 64,
        output_dir: Union[str, Path] = "data/processed",
    ):
        self.dataset_name = dataset_name.upper()
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride_train = stride_train
        self.stride_test = stride_test
        self.val_ratio = val_ratio
        self.scale_method = scale_method
        self.drop_constants = drop_constants
        self.label_policy = LabelPolicy(label_policy)
        self.batch_size = batch_size
        self.output_dir = Path(output_dir) / dataset_name.lower()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = IndustrialScaler(method=scale_method)
        self.sensor_columns: List[str] = []
        self.dropped_constants: List[str] = []
        self.artifacts: Optional[PipelineArtifacts] = None

    def run(
        self,
        normal_max_rows: Optional[int] = None,
        attack_max_rows: Optional[int] = None,
    ) -> Tuple[SlidingWindowDataset, SlidingWindowDataset, SlidingWindowDataset]:
        """
        Executes the entire zero-leakage pipeline.

        Returns:
            train_dataset: Normal training sliding windows.
            val_dataset: Normal validation sliding windows.
            test_dataset: Attack/test sliding windows with labels.
        """
        print(f"[{self.dataset_name}] 1. Loading raw telemetry from {self.data_dir}...")
        if "WADI" in self.dataset_name:
            df_norm_raw, df_att_raw, initial_sensors = load_raw_wadi(
                self.data_dir,
                normal_max_rows=normal_max_rows,
                attack_max_rows=attack_max_rows,
            )
        elif "SWAT" in self.dataset_name:
            df_norm_raw, df_att_raw, initial_sensors = load_raw_swat(
                self.data_dir,
                rows_per_file=normal_max_rows,
            )
        else:
            raise ValueError(f"Unknown dataset '{self.dataset_name}', expected 'WADI' or 'SWAT'")

        # 2. Chronological Split (Train vs Val)
        print(f"[{self.dataset_name}] 2. Chronological splitting (val_ratio={self.val_ratio})...")
        df_train_raw, df_val_raw = split_chronological_normal(
            df_norm_raw, val_ratio=self.val_ratio
        )

        # 3. Handle Missing Values (Train-Only Medians)
        print(f"[{self.dataset_name}] 3. Imputing missing values using train medians...")
        df_train, train_medians = handle_missing_values(df_train_raw, initial_sensors)
        df_val, _ = handle_missing_values(df_val_raw, initial_sensors, train_medians=train_medians)
        df_test, _ = handle_missing_values(df_att_raw, initial_sensors, train_medians=train_medians)

        # 4. Detect & Filter Constant Columns on Train
        active_sensors = list(initial_sensors)
        if self.drop_constants:
            print(f"[{self.dataset_name}] 4. Checking constant sensor columns on train split...")
            constant_cols = detect_constant_columns(df_train, active_sensors)
            self.dropped_constants = constant_cols
            if constant_cols:
                print(f"    Dropping {len(constant_cols)} constant features: {constant_cols[:5]}...")
                active_sensors = [c for c in active_sensors if c not in constant_cols]
        self.sensor_columns = active_sensors

        # 5. Fit Scaler ONLY on Training Split
        print(f"[{self.dataset_name}] 5. Fitting {self.scale_method} scaler strictly on train normal data...")
        self.scaler.fit(df_train, self.sensor_columns)
        scaler_file = self.output_dir / "scaler.json"
        self.scaler.save(scaler_file)

        # Transform splits
        X_train = self.scaler.transform(df_train)
        X_val = self.scaler.transform(df_val)
        X_test = self.scaler.transform(df_test)

        y_train = np.zeros(len(X_train), dtype=np.int64)
        y_val = np.zeros(len(X_val), dtype=np.int64)
        y_test = df_test["label"].to_numpy(dtype=np.int64)

        # 6. Build Window Datasets
        print(f"[{self.dataset_name}] 6. Creating sliding window datasets (L={self.window_size}, S_train={self.stride_train}, S_test={self.stride_test})...")
        train_dataset = SlidingWindowDataset(
            X_train, y_train, window_size=self.window_size, stride=self.stride_train, label_policy=self.label_policy
        )
        val_dataset = SlidingWindowDataset(
            X_val, y_val, window_size=self.window_size, stride=self.stride_train, label_policy=self.label_policy
        )
        test_dataset = SlidingWindowDataset(
            X_test, y_test, window_size=self.window_size, stride=self.stride_test, label_policy=self.label_policy
        )

        # Calculate test anomaly ratio
        test_labels = [test_dataset[i][1].item() if test_dataset[i][1].ndim == 0 else test_dataset[i][1].any().item() for i in range(len(test_dataset))]
        anomaly_ratio = float(np.mean(test_labels)) if test_labels else 0.0

        # Save metadata and contract
        self.artifacts = PipelineArtifacts(
            dataset_name=self.dataset_name,
            sensor_columns=self.sensor_columns,
            dropped_constant_columns=self.dropped_constants,
            num_train_timesteps=len(X_train),
            num_val_timesteps=len(X_val),
            num_test_timesteps=len(X_test),
            num_train_windows=len(train_dataset),
            num_val_windows=len(val_dataset),
            num_test_windows=len(test_dataset),
            test_anomaly_ratio=round(anomaly_ratio, 4),
            window_size=self.window_size,
            stride_train=self.stride_train,
            stride_test=self.stride_test,
            label_policy=self.label_policy.value,
            scaler_path=str(scaler_file),
        )

        metadata_file = self.output_dir / "pipeline_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(asdict(self.artifacts), f, indent=2)

        print(f"[{self.dataset_name}] [OK] Pipeline complete. Windows: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        print(f"    Features: {len(self.sensor_columns)} active sensors (dropped {len(self.dropped_constants)} constants)")
        print(f"    Test window anomaly ratio: {self.artifacts.test_anomaly_ratio * 100:.2f}%")
        print(f"    Artifacts saved to {self.output_dir}")

        return train_dataset, val_dataset, test_dataset

    def get_dataloaders(
        self,
        train_dataset: SlidingWindowDataset,
        val_dataset: SlidingWindowDataset,
        test_dataset: SlidingWindowDataset,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates PyTorch DataLoaders.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle windows during training
            num_workers=num_workers,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        return train_loader, val_loader, test_loader


__all__ = [
    "PipelineArtifacts",
    "TelemetryPipeline",
]
