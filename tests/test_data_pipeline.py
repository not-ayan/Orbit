"""
Unit and integration tests for ORBIT leakage-safe data pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from orbit.data.preprocessing import (
    split_chronological_normal,
    handle_missing_values,
    detect_constant_columns,
    IndustrialScaler,
)
from orbit.data.windowing import (
    SlidingWindowDataset,
    LabelPolicy,
    extract_sliding_windows_numpy,
)
from orbit.data.pipeline import TelemetryPipeline


@pytest.fixture
def synthetic_telemetry():
    """Generates synthetic multivariate telemetry for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Continuous sine waves + noise
    t = np.linspace(0, 50, n_samples)
    data = np.zeros((n_samples, n_features))
    for i in range(n_features - 2):
        data[:, i] = np.sin(t * (i + 1) * 0.1) + np.random.normal(0, 0.05, n_samples)

    # 1 constant feature
    data[:, -2] = 5.0
    # 1 feature with initial missing values
    data[:, -1] = np.cos(t)
    data[:5, -1] = np.nan

    sensor_names = [f"sensor_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=sensor_names)
    df["timestamp"] = pd.date_range("2026-01-01", periods=n_samples, freq="1s")
    df["label"] = 0

    # Inject a known anomaly burst in the last 100 samples
    df.loc[900:920, "sensor_0"] += 10.0
    df.loc[900:920, "label"] = 1

    return df, sensor_names


def test_chronological_split(synthetic_telemetry):
    df, _ = synthetic_telemetry
    val_ratio = 0.2
    df_train, df_val = split_chronological_normal(df, val_ratio=val_ratio)

    assert len(df_train) == int(len(df) * 0.8)
    assert len(df_val) == int(len(df) * 0.2)
    assert len(df_train) + len(df_val) == len(df)

    # Strict temporal boundary check: max train timestamp < min val timestamp
    assert df_train["timestamp"].max() < df_val["timestamp"].min()


def test_missing_value_imputation(synthetic_telemetry):
    df, sensor_names = synthetic_telemetry
    df_train, df_val = split_chronological_normal(df, val_ratio=0.2)

    df_train_imp, medians = handle_missing_values(df_train, sensor_names)
    assert not df_train_imp[sensor_names].isnull().any().any()

    # Impute validation using the TRAIN medians
    df_val_imp, _ = handle_missing_values(df_val, sensor_names, train_medians=medians)
    assert not df_val_imp[sensor_names].isnull().any().any()


def test_constant_column_detection(synthetic_telemetry):
    df, sensor_names = synthetic_telemetry
    df_train, _ = split_chronological_normal(df, val_ratio=0.2)
    constants = detect_constant_columns(df_train, sensor_names)

    assert "sensor_8" in constants
    assert "sensor_0" not in constants


def test_industrial_scaler_zero_leakage(synthetic_telemetry):
    df, sensor_names = synthetic_telemetry
    df_train, df_val = split_chronological_normal(df, val_ratio=0.2)

    active_sensors = [c for c in sensor_names if c != "sensor_8"]
    scaler = IndustrialScaler(method="standard")

    # Fit strictly on train
    scaler.fit(df_train, active_sensors)

    # Check fitted mean matches train mean, NOT overall mean
    expected_train_mean = df_train[active_sensors].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_train_mean, rtol=1e-3, atol=1e-3)

    X_train_scaled = scaler.transform(df_train)
    X_val_scaled = scaler.transform(df_val)

    # Train scaled should have near zero mean
    np.testing.assert_allclose(np.mean(X_train_scaled, axis=0), 0.0, atol=1e-2)

    # Save and reload test
    test_json = Path("tests/temp_scaler.json")
    scaler.save(test_json)
    loaded_scaler = IndustrialScaler.load(test_json)
    np.testing.assert_allclose(scaler.mean_, loaded_scaler.mean_)
    np.testing.assert_allclose(scaler.std_, loaded_scaler.std_)
    test_json.unlink(missing_ok=True)


def test_sliding_window_dataset_shapes_and_policies():
    T, F = 500, 8
    L, S = 128, 32
    data = np.random.randn(T, F).astype(np.float32)
    labels = np.zeros(T, dtype=np.int64)
    # Put anomaly in range 130 to 140
    labels[130:140] = 1

    # ANY policy
    ds_any = SlidingWindowDataset(data, labels, window_size=L, stride=S, label_policy=LabelPolicy.ANY)
    expected_windows = (T - L) // S + 1
    assert len(ds_any) == expected_windows

    x, y = ds_any[0]
    assert x.shape == (L, F)
    assert isinstance(x, torch.Tensor)
    assert y.item() == 0  # Window 0 is [0:128], anomaly is at 130

    x1, y1 = ds_any[1]  # Window 1 is [32:160], encompasses [130:140]
    assert y1.item() == 1  # Contains anomaly!

    # PER_TIMESTEP policy
    ds_seq = SlidingWindowDataset(data, labels, window_size=L, stride=S, label_policy=LabelPolicy.PER_TIMESTEP)
    x_s, y_s = ds_seq[1]
    assert y_s.shape == (L,)
    assert y_s.sum() == 10  # 10 anomalous timesteps


def test_numpy_sliding_window_vectorized():
    T, F = 200, 4
    L, S = 50, 25
    data = np.arange(T * F, dtype=np.float32).reshape(T, F)
    labels = np.zeros(T, dtype=np.int64)
    labels[60] = 1

    X, y = extract_sliding_windows_numpy(data, labels, window_size=L, stride=S, label_policy="any")
    expected_windows = (T - L) // S + 1
    assert X.shape == (expected_windows, L, F)
    assert y.shape == (expected_windows,)
    assert y[0] == 0  # [0:50]
    assert y[1] == 1  # [25:75] contains index 60!

