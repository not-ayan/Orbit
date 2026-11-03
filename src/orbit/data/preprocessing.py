"""
Leakage-safe preprocessing and scaling for industrial time-series telemetry.
Rules:
1. Scalers must be fit ONLY on training normal data.
2. Missing values handled with temporal forward-fill followed by train medians.
3. Chronological splitting strictly enforced before windowing.
"""

from typing import Tuple, List, Dict, Optional, Union
import json
from pathlib import Path
import numpy as np
import pandas as pd


def split_chronological_normal(
    df_normal: pd.DataFrame,
    val_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits normal telemetry chronologically into training and validation sets.
    Preserves strict temporal order: Train comes before Validation.

    Args:
        df_normal: Cleaned normal operation DataFrame.
        val_ratio: Fraction of normal telemetry reserved for validation (e.g. 0.2 = 20%).

    Returns:
        df_train: Early chronological portion.
        df_val: Later chronological portion.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    n_rows = len(df_normal)
    split_idx = int(n_rows * (1.0 - val_ratio))

    df_train = df_normal.iloc[:split_idx].copy().reset_index(drop=True)
    df_val = df_normal.iloc[split_idx:].copy().reset_index(drop=True)

    return df_train, df_val


def handle_missing_values(
    df: pd.DataFrame,
    sensor_columns: List[str],
    train_medians: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handles missing sensor telemetry using physically sound imputation:
    1. Forward-fill (sensors maintain last measured physical state).
    2. Backward-fill / Train-median fill for missing initial timestamps.

    Args:
        df: DataFrame containing telemetry.
        sensor_columns: List of numeric sensor columns.
        train_medians: Precomputed medians from training split. If None, computed from df.

    Returns:
        df_imputed: Imputed DataFrame.
        medians: Medians used for initial values (fitted on training data).
    """
    df = df.copy()

    # Forward fill physical sensor measurements
    df[sensor_columns] = df[sensor_columns].ffill()

    if train_medians is None:
        train_medians = df[sensor_columns].median(numeric_only=True)

    # Impute any leading NaNs using train medians
    df[sensor_columns] = df[sensor_columns].fillna(train_medians)
    # If any still remain (e.g. median was NaN), fill with 0.0 as final fallback
    df[sensor_columns] = df[sensor_columns].fillna(0.0)

    return df, train_medians


def detect_constant_columns(
    df_train: pd.DataFrame,
    sensor_columns: List[str],
    variance_threshold: float = 1e-8,
) -> List[str]:
    """
    Identifies constant or near-zero variance sensor channels in the training split.
    """
    variances = df_train[sensor_columns].var(numeric_only=True)
    constant_cols = list(variances[variances <= variance_threshold].index)
    return constant_cols


class IndustrialScaler:
    """
    StandardScaler or MinMaxScaler designed specifically for zero-leakage
    industrial telemetry workflows. Fits strictly on training normal data,
    saves parameters to JSON, and transforms test sets deterministically.
    """

    def __init__(self, method: str = "standard", clip_outliers: Optional[float] = 10.0):
        if method not in ("standard", "minmax"):
            raise ValueError(f"Method must be 'standard' or 'minmax', got '{method}'")
        self.method = method
        self.clip_outliers = clip_outliers
        self.feature_names_: List[str] = []
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None

    def fit(
        self,
        df_train: Union[pd.DataFrame, np.ndarray],
        sensor_columns: Optional[List[str]] = None,
    ) -> "IndustrialScaler":
        """
        Fit scaling parameters strictly on training normal data.
        Supports both pandas DataFrame and numpy ndarray.
        """
        if isinstance(df_train, np.ndarray):
            X = df_train.astype(np.float32)
            self.feature_names_ = list(sensor_columns) if sensor_columns else [f"f_{i}" for i in range(X.shape[1])]
        else:
            if sensor_columns is None:
                raise ValueError("sensor_columns must be provided when fitting on a DataFrame")
            self.feature_names_ = list(sensor_columns)
            X = df_train[sensor_columns].to_numpy(dtype=np.float32)

        if self.method == "standard":
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0)
            # Avoid division by zero for constant features
            self.std_[self.std_ < 1e-6] = 1.0
        elif self.method == "minmax":
            self.min_ = np.nanmin(X, axis=0)
            self.max_ = np.nanmax(X, axis=0)
            diff = self.max_ - self.min_
            diff[diff < 1e-6] = 1.0
            self.scale_diff_ = diff

        return self

    def transform(self, df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transforms input DataFrame or ndarray using pre-fitted training statistics.
        """
        if self.mean_ is None and self.min_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")

        if isinstance(df, np.ndarray):
            X = df.astype(np.float32)
        else:
            X = df[self.feature_names_].to_numpy(dtype=np.float32)

        if self.method == "standard":
            X_scaled = (X - self.mean_) / self.std_
            if self.clip_outliers is not None:
                X_scaled = np.clip(X_scaled, -self.clip_outliers, self.clip_outliers)
        elif self.method == "minmax":
            X_scaled = (X - self.min_) / self.scale_diff_

        # Replace any remaining NaNs or Infs
        np.nan_to_num(X_scaled, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
        return X_scaled

    def fit_transform(self, df_train: pd.DataFrame, sensor_columns: List[str]) -> np.ndarray:
        return self.fit(df_train, sensor_columns).transform(df_train)

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "clip_outliers": self.clip_outliers,
            "feature_names": self.feature_names_,
            "mean": self.mean_.tolist() if self.mean_ is not None else None,
            "std": self.std_.tolist() if self.std_ is not None else None,
            "min": self.min_.tolist() if self.min_ is not None else None,
            "max": self.max_.tolist() if self.max_ is not None else None,
        }

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "IndustrialScaler":
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        scaler = cls(method=data["method"], clip_outliers=data.get("clip_outliers", 10.0))
        scaler.feature_names_ = data["feature_names"]
        if data.get("mean") is not None:
            scaler.mean_ = np.array(data["mean"], dtype=np.float32)
        if data.get("std") is not None:
            scaler.std_ = np.array(data["std"], dtype=np.float32)
        if data.get("min") is not None:
            scaler.min_ = np.array(data["min"], dtype=np.float32)
        if data.get("max") is not None:
            scaler.max_ = np.array(data["max"], dtype=np.float32)
            diff = scaler.max_ - scaler.min_
            diff[diff < 1e-6] = 1.0
            scaler.scale_diff_ = diff
        return scaler


__all__ = [
    "split_chronological_normal",
    "handle_missing_values",
    "detect_constant_columns",
    "IndustrialScaler",
]

