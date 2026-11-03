"""
Dataset schema definitions and column specifications for industrial telemetry.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class DatasetSchema:
    name: str
    time_columns: List[str]
    label_column: Optional[str]
    metadata_columns: List[str]
    known_invalid_columns: List[str]
    label_mapping: Dict[any, int]  # Maps raw label value -> binary (0: Normal, 1: Anomaly)
    sampling_rate_hz: float = 1.0


# WADI (Water Distribution Testbed) - A2 (2019 Release)
WADI_A2_ALL_NULL_COLUMNS = [
    "2_LS_001_AL",
    "2_LS_002_AL",
    "2_P_001_STATUS",
    "2_P_002_STATUS",
]

WADI_A2_SCHEMA = DatasetSchema(
    name="WADI_A2",
    time_columns=["Date", "Time"],
    label_column="Attack LABLE (1:No Attack, -1:Attack)",
    metadata_columns=["Row", "Date", "Time"],
    known_invalid_columns=WADI_A2_ALL_NULL_COLUMNS,
    label_mapping={
        1: 0,    # 1 indicates Normal in raw WADI
        -1: 1,   # -1 indicates Attack in raw WADI
        "1": 0,
        "-1": 1,
    },
    sampling_rate_hz=1.0,
)

# SWaT (Secure Water Treatment Testbed) - A8 (June-July 2021)
SWAT_A8_METADATA_COLUMNS = [
    "Timestamp",
    "Annotation",
    "Other Anomalies",
    "A#AEGIS",
    "A#AICrit",
    "A#Attester",
    "A#DAD",
    "A#HybMon",
    "A#Stef-Phy",
    "A#Superdetector",
    "Attack Hash",
    "Attack Name",
    "Attack State",
    "Attack Target",
    "Attack Type",
    "Intent",
    "Attack Mode",
    "Attack Outcome",
    "Target Selection",
    "Entry Point",
    "ASD",
    "Attacker",
    "Attack Id",
    "Attack Subid",
]

SWAT_A8_SCHEMA = DatasetSchema(
    name="SWaT_A8",
    time_columns=["Timestamp"],
    label_column="Other Anomalies",
    metadata_columns=SWAT_A8_METADATA_COLUMNS,
    known_invalid_columns=[],
    label_mapping={},  # null -> 0, not-null -> 1 handled in loading
    sampling_rate_hz=1.0,
)

__all__ = [
    "DatasetSchema",
    "WADI_A2_SCHEMA",
    "WADI_A2_ALL_NULL_COLUMNS",
    "SWAT_A8_SCHEMA",
    "SWAT_A8_METADATA_COLUMNS",
]

