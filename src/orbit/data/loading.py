"""
Data loading utilities for industrial telemetry datasets (SWaT, WADI).
"""

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np

from orbit.data.schema import (
    WADI_A2_SCHEMA,
    WADI_A2_ALL_NULL_COLUMNS,
    SWAT_A8_SCHEMA,
    SWAT_A8_METADATA_COLUMNS,
)


def load_raw_wadi(
    wadi_dir: Path,
    normal_max_rows: Optional[int] = None,
    attack_max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads raw WADI A2 telemetry.

    Returns:
        df_normal: DataFrame of normal telemetry with 'label' = 0.
        df_attack: DataFrame of attack telemetry with 'label' in {0, 1}.
        sensor_columns: List of valid numeric sensor feature column names.
    """
    wadi_dir = Path(wadi_dir)
    normal_csv = wadi_dir / "WADI_14days_new.csv"
    attack_csv = wadi_dir / "WADI_attackdataLABLE.csv"

    if not normal_csv.exists() or not attack_csv.exists():
        # Check if nested
        nested_dir = wadi_dir / "WADI.A2_19 Nov 2019"
        if nested_dir.exists():
            normal_csv = nested_dir / "WADI_14days_new.csv"
            attack_csv = nested_dir / "WADI_attackdataLABLE.csv"

    if not normal_csv.exists():
        raise FileNotFoundError(f"WADI normal CSV not found in {wadi_dir}")
    if not attack_csv.exists():
        raise FileNotFoundError(f"WADI attack CSV not found in {wadi_dir}")

    # Read normal data
    df_normal = pd.read_csv(normal_csv, nrows=normal_max_rows).copy()
    df_normal.columns = [c.strip() for c in df_normal.columns]
    df_normal["label"] = 0

    # Read attack data: line 0 in raw file is index (0,1,2...), line 1 is column headers
    df_attack = pd.read_csv(attack_csv, skiprows=1, nrows=attack_max_rows).copy()
    df_attack.columns = [c.strip() for c in df_attack.columns]

    # Map attack label: 1 -> 0 (Normal), -1 -> 1 (Anomaly)
    raw_label_col = WADI_A2_SCHEMA.label_column
    if raw_label_col not in df_attack.columns:
        matching = [c for c in df_attack.columns if "Attack" in c or "LABLE" in c]
        if matching:
            raw_label_col = matching[0]
        else:
            raise KeyError(f"Could not find label column '{raw_label_col}' in WADI attack data")

    df_attack["label"] = df_attack[raw_label_col].map(WADI_A2_SCHEMA.label_mapping).fillna(0).astype(int)

    # Identify sensor columns
    ignore_cols = set(WADI_A2_SCHEMA.metadata_columns + [raw_label_col, "label"] + WADI_A2_ALL_NULL_COLUMNS)
    sensor_columns = [c for c in df_normal.columns if c not in ignore_cols]

    # Drop entirely null columns from both
    drop_cols = [c for c in WADI_A2_ALL_NULL_COLUMNS if c in df_normal.columns]
    if drop_cols:
        df_normal = df_normal.drop(columns=drop_cols)
    drop_cols_att = [c for c in WADI_A2_ALL_NULL_COLUMNS if c in df_attack.columns]
    if drop_cols_att:
        df_attack = df_attack.drop(columns=drop_cols_att)

    # Coerce sensor columns to numeric
    for c in sensor_columns:
        if c in df_normal.columns:
            df_normal[c] = pd.to_numeric(df_normal[c], errors="coerce")
        if c in df_attack.columns:
            df_attack[c] = pd.to_numeric(df_attack[c], errors="coerce")

    return df_normal, df_attack, sensor_columns


def load_raw_swat(
    swat_dir: Path,
    max_files: Optional[int] = None,
    rows_per_file: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Loads raw SWaT A8 telemetry across daily CSV files.
    Files before 2021-07-12 are continuous normal operation.
    Files from 2021-07-12 onwards contain physical attack tests.

    Returns:
        df_normal: DataFrame of normal telemetry with 'label' = 0.
        df_attack: DataFrame of attack telemetry with 'label' in {0, 1}.
        sensor_columns: List of valid numeric sensor feature column names.
    """
    swat_dir = Path(swat_dir)
    csv_files = sorted(list(swat_dir.glob("**/*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No SWaT CSV files found in {swat_dir}")

    if max_files is not None:
        csv_files = csv_files[:max_files]

    dfs_normal = []
    dfs_attack = []
    detected_sensor_cols = None

    for f in csv_files:
        df_file = pd.read_csv(f, nrows=rows_per_file)
        df_file.columns = [c.strip() for c in df_file.columns]

        # Determine sensor columns from first file
        if detected_sensor_cols is None:
            ignore_set = set(SWAT_A8_METADATA_COLUMNS + ["label"])
            detected_sensor_cols = [c for c in df_file.columns if c not in ignore_set]

        # Anomaly label: 1 if Other Anomalies or Annotation is non-empty/non-null
        is_anomaly = pd.Series(0, index=df_file.index, dtype=int)
        if "Other Anomalies" in df_file.columns:
            is_anomaly = is_anomaly | df_file["Other Anomalies"].notna().astype(int)
        if "Annotation" in df_file.columns:
            is_anomaly = is_anomaly | df_file["Annotation"].notna().astype(int)

        df_file["label"] = is_anomaly

        # Chronological distinction: files before July 12 2021 are clean normal operation
        if "20210712" in f.name or "20210714" in f.name or "20210721" in f.name or is_anomaly.sum() > 0:
            dfs_attack.append(df_file)
        else:
            dfs_normal.append(df_file)

    df_normal = pd.concat(dfs_normal, ignore_index=True) if dfs_normal else pd.DataFrame()
    df_attack = pd.concat(dfs_attack, ignore_index=True) if dfs_attack else pd.DataFrame()

    # Coerce sensor columns to numeric (turning SCADA strings like 'Bad Input' into NaN)
    for c in detected_sensor_cols:
        if c in df_normal.columns:
            df_normal[c] = pd.to_numeric(df_normal[c], errors="coerce")
        if c in df_attack.columns:
            df_attack[c] = pd.to_numeric(df_attack[c], errors="coerce")

    return df_normal, df_attack, detected_sensor_cols


__all__ = [
    "load_raw_wadi",
    "load_raw_swat",
]
