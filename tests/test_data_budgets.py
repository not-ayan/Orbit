"""
Unit and audit tests for target-domain data budget accounting.
Guarantees strict data budgets, non-overlapping chronological splits, and zero test label leakage.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from orbit.adaptation.transfer import (
    BudgetProtocol,
    BudgetAllocation,
    allocate_target_budget,
)
from orbit.data.windowing import SlidingWindowDataset


def create_mock_target_dataset(num_samples: int = 500, window_size: int = 50, stride: int = 10):
    data = np.arange(num_samples)[:, None].repeat(5, axis=1).astype(np.float32)
    labels = np.zeros(num_samples, dtype=np.int64)
    # Inject synthetic anomalies only in the second half
    labels[400:] = 1
    dataset = SlidingWindowDataset(data, labels=labels, window_size=window_size, stride=stride)
    return dataset


def test_budget_allocation_zero_shot():
    dataset = create_mock_target_dataset()
    alloc = allocate_target_budget(dataset, budget_percent=0.0)

    assert alloc.train_subset is None
    assert alloc.val_subset is None
    assert alloc.cal_subset is None
    assert alloc.metadata["total_budgeted_windows"] == 0
    assert alloc.metadata["num_train_windows"] == 0


def test_budget_allocation_strict_chronological_splits():
    dataset = create_mock_target_dataset(num_samples=1000, window_size=100, stride=10)
    # total windows: (1000 - 100) // 10 + 1 = 91 windows

    # Test 10% budget
    alloc_10 = allocate_target_budget(dataset, budget_percent=10.0, protocol=BudgetProtocol.STRICT_INCLUSIVE)
    meta = alloc_10.metadata

    assert meta["total_budgeted_windows"] == 10
    assert meta["num_train_windows"] + meta["num_val_windows"] + meta["num_cal_windows"] == 10

    # Verify strictly non-overlapping and chronological indices
    train_idx = alloc_10.train_subset.indices
    val_idx = alloc_10.val_subset.indices
    cal_idx = alloc_10.cal_subset.indices

    assert max(train_idx) < min(val_idx)
    assert max(val_idx) < min(cal_idx)


def test_budget_allocation_minimal_single_window():
    dataset = create_mock_target_dataset()
    alloc = allocate_target_budget(dataset, budget_percent=1.0)

    assert alloc.metadata["total_budgeted_windows"] >= 1
    assert alloc.train_subset is not None
    # Verify strict no-duplication policy
    train_idx = set(alloc.train_subset.indices)
    cal_idx = set(alloc.cal_subset.indices if alloc.cal_subset else [])
    assert len(train_idx.intersection(cal_idx)) == 0


def test_test_label_leakage_protection():
    """
    Verifies that target test labels are completely isolated from adaptation and calibration.
    """
    dataset = create_mock_target_dataset()
    alloc = allocate_target_budget(dataset, budget_percent=5.0)

    # Verify that metadata explicitly tracks label isolation
    assert alloc.metadata["test_labels_accessed_during_training"] is False
    assert alloc.metadata["test_labels_accessed_during_calibration"] is False

    # Check that labels in training subset are strictly normal (0)
    train_loader = DataLoader(alloc.train_subset, batch_size=2)
    for _, batch_y in train_loader:
        assert (batch_y == 0).all()


def test_allocate_raw_target_budget_disjoint_and_embargo():
    """
    Verifies that raw timeline partitioning yields pairwise disjoint sample ranges
    separated by >= window_size embargo gaps, and fits scaler strictly on train.
    """
    from orbit.adaptation.transfer import allocate_raw_target_budget

    T, F = 6000, 8
    L, S = 100, 20
    raw_data = np.random.randn(T, F).astype(np.float32)

    # 10% budget: 600 samples, plenty of room for train, embargo (100), val, embargo (100), cal
    alloc = allocate_raw_target_budget(
        raw_normal_data=raw_data,
        budget_percent=10.0,
        window_size=L,
        stride=S,
        embargo_steps=L,
    )
    manifest = alloc.manifest
    train_range = manifest["train_raw_range"]
    val_range = manifest["val_raw_range"]
    cal_range = manifest["cal_raw_range"]

    # Verify chronological order and embargo gaps
    assert train_range[1] + L <= val_range[0]
    assert val_range[1] + L <= cal_range[0]
    assert cal_range[1] <= manifest["total_budgeted_raw_rows"]

    # Verify scaler was fit strictly on train_range
    expected_mean = raw_data[train_range[0]:train_range[1]].mean(axis=0)
    np.testing.assert_allclose(alloc.scaler.mean_, expected_mean, atol=1e-5)

    # Verify changing held-out test data has 0% effect on fitted scaler
    held_out_test_data = np.random.randn(500, F) * 50.0
    _ = alloc.scaler.transform(held_out_test_data)
    np.testing.assert_allclose(alloc.scaler.mean_, expected_mean, atol=1e-5)


def test_no_window_duplication_tiny_budget():
    """
    Verifies that at tiny budgets (1 window), no window is duplicated across partitions.
    """
    dataset = create_mock_target_dataset(num_samples=200, window_size=50, stride=50)
    # small number of windows
    alloc = allocate_target_budget(dataset, budget_percent=1.0)
    meta = alloc.metadata

    # Check window indices are disjoint
    train_idx = set(alloc.train_subset.indices if alloc.train_subset else [])
    val_idx = set(alloc.val_subset.indices if alloc.val_subset else [])
    cal_idx = set(alloc.cal_subset.indices if alloc.cal_subset else [])

    assert len(train_idx.intersection(val_idx)) == 0
    assert len(train_idx.intersection(cal_idx)) == 0
    assert len(val_idx.intersection(cal_idx)) == 0

