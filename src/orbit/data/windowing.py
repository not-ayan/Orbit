"""
Sliding window generation and label alignment for multivariate time series.
Supports both zero-copy PyTorch Dataset and in-memory vectorized window extraction.
"""

from enum import Enum
from typing import Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class LabelPolicy(str, Enum):
    ANY = "any"                 # 1 if ANY timestamp in window is anomalous
    MAJORITY = "majority"       # 1 if >50% of timestamps in window are anomalous
    CENTER = "center"           # Label at window midpoint L // 2
    LAST = "last"               # Label at window endpoint L - 1
    PER_TIMESTEP = "per_step"   # Full sequence of per-timestep labels (L,)


class SlidingWindowDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for continuous multivariate time series.
    Avoids copying/duplicating multi-gigabyte overlapping arrays into memory.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 128,
        stride: int = 32,
        label_policy: Union[str, LabelPolicy] = LabelPolicy.ANY,
    ):
        """
        Args:
            data: 2D array of shape (T, F) representing normalized telemetry.
            labels: 1D array of shape (T,) with binary labels (0: Normal, 1: Anomaly).
            window_size: Sequence length L (default 128).
            stride: Stride S between consecutive window starts (default 32).
            label_policy: Aggregation strategy for window label.
        """
        if data.ndim != 2:
            raise ValueError(f"Expected data to be 2D (T, F), got shape {data.shape}")

        self.data = torch.from_numpy(data.astype(np.float32))
        self.window_size = window_size
        self.stride = stride
        self.label_policy = LabelPolicy(label_policy) if isinstance(label_policy, str) else label_policy

        if labels is not None:
            if len(labels) != len(data):
                raise ValueError(f"Data length {len(data)} != Labels length {len(labels)}")
            self.labels = torch.from_numpy(labels.astype(np.int64))
        else:
            self.labels = None

        self.num_windows = max(0, (len(self.data) - self.window_size) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.window_size

        x = self.data[start:end]  # Shape: (L, F)

        if self.labels is None:
            y = torch.tensor(0, dtype=torch.long)
            return x, y

        y_window = self.labels[start:end]

        if self.label_policy == LabelPolicy.ANY:
            y = (y_window.sum() > 0).long()
        elif self.label_policy == LabelPolicy.MAJORITY:
            y = (y_window.sum() > (self.window_size // 2)).long()
        elif self.label_policy == LabelPolicy.CENTER:
            y = y_window[self.window_size // 2]
        elif self.label_policy == LabelPolicy.LAST:
            y = y_window[-1]
        elif self.label_policy == LabelPolicy.PER_TIMESTEP:
            y = y_window

        return x, y


def extract_sliding_windows_numpy(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    window_size: int = 128,
    stride: int = 32,
    label_policy: Union[str, LabelPolicy] = LabelPolicy.ANY,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Vectorized extraction of sliding window numpy arrays using numpy stride tricks.
    Returns:
        X: (num_windows, window_size, num_features) float32 array
        y: (num_windows,) or (num_windows, window_size) int64 array
    """
    policy = LabelPolicy(label_policy) if isinstance(label_policy, str) else label_policy
    n_samples, n_features = data.shape
    num_windows = (n_samples - window_size) // stride + 1

    if num_windows <= 0:
        return np.empty((0, window_size, n_features), dtype=data.dtype), None

    # Stride tricks for zero-copy 3D view
    shape = (num_windows, window_size, n_features)
    strides = (data.strides[0] * stride, data.strides[0], data.strides[1])
    X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

    if labels is None:
        return X, None

    shape_y = (num_windows, window_size)
    strides_y = (labels.strides[0] * stride, labels.strides[0])
    y_windows = np.lib.stride_tricks.as_strided(labels, shape=shape_y, strides=strides_y)

    if policy == LabelPolicy.ANY:
        y = (y_windows.sum(axis=1) > 0).astype(np.int64)
    elif policy == LabelPolicy.MAJORITY:
        y = (y_windows.sum(axis=1) > (window_size // 2)).astype(np.int64)
    elif policy == LabelPolicy.CENTER:
        y = y_windows[:, window_size // 2].astype(np.int64)
    elif policy == LabelPolicy.LAST:
        y = y_windows[:, -1].astype(np.int64)
    elif policy == LabelPolicy.PER_TIMESTEP:
        y = y_windows.astype(np.int64)

    return X, y


__all__ = [
    "LabelPolicy",
    "SlidingWindowDataset",
    "extract_sliding_windows_numpy",
]

