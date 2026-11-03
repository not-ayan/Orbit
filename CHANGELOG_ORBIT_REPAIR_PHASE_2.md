# ORBIT Repair Phase 2 — Changelog

**Date**: 2026-11-03  
**Status**: All Findings A–G Resolved & Validated across 48 Unit, Integration, and Regression Tests  
**Execution Pack Reference**: `ORBIT_Second_Repair_Pack` (Tasks 0 to 7)

---

## 1. Re-Audit & Defect Reproduction (Task 0)

Every reported issue from the review was reproduced against the live codebase:
- **Finding A.1 (Tensor Boolean Ambiguity)**: Confirmed `source_in_w = state.get(...) or state.get(...)` throws `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`.
- **Finding A.4 (Sensor Mapping Idle)**: Confirmed `sensor_mapping` was saved in `__init__` but omitted from `forward()`.
- **Finding A.5 (Bypassing Inference Validation)**: Confirmed `evaluator.py:compute_step_errors` called `model(batch_x)` directly, bypassing `validate_for_inference()`.
- **Finding B.1 & B.2 (Scaler Pre-Fitting & Window Overlap)**: Confirmed `IndustrialScaler` was fit on the full 80% train normal pool before budget allocation, and sliding windows shared raw observations across split boundaries.
- **Finding C (Threshold vs Score Scale Inconsistency)**: Confirmed threshold was calibrated on window-average MSE but applied to continuous timestamp scores with different variances.
- **Finding D (Event Metric Inflation)**: Confirmed `compute_event_metrics` permitted multiple alarm episodes to match the same ground truth event.
- **Finding E (0% Crash & Untrained Models)**: Confirmed budget 0% passed `None` calibration loader to evaluator and instantiated untrained adapters.
- **Finding F (Test Portability)**: Confirmed `test_resolve_dataset_dir_known` failed in environments without raw SWaT telemetry folders.
- **Finding G (Missing LSTM Few-Shot Baseline)**: Confirmed LSTM Autoencoder was omitted from the few-shot transfer benchmark.

---

## 2. Implemented Code Repairs

### Task 1: Complete Checkpoint State & Safe Zero-Shot Inference
- **Ambiguous Truth-Value Fix**: Replaced Python `or` operators on tensors with explicit `is None` checks in [`src/orbit/adaptation/adapter.py`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/adapter.py#L218-L230).
- **Functional Sensor Mapping**: Implemented active sensor mapping execution in [`AdaptedTransformerModel.forward`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/adapter.py#L340-L380):
  - Projects target input $X_{\text{target}}$ to source space $X_{\text{source}}$.
  - Runs through trained source projection, positional encoding, backbone, and reconstruction head.
  - Maps reconstructed source output back to target telemetry dimensions.
- **Mandatory Inference Validation**:
  - Enforced `self.validate_for_inference()` in `AdaptedTransformerModel.forward` for `ZERO_SHOT` mode.
  - Enforced `model.validate_for_inference()` at the start of `AnomalyEvaluator.compute_step_errors`.
- **Normalization Stats in Checkpoint**: Added `normalization_stats` bundling in [`save_orbit_checkpoint`](file:///F:/ORBIT_Dataset/src/orbit/models/checkpoint.py#L21).

### Task 2: Raw-Time Separation & Leakage-Safe Target Budgets
- **Raw-Timeline Partitioning with Embargo Gaps**: Implemented [`allocate_raw_target_budget`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/transfer.py#L58):
  - Slices raw telemetry chronologically into `train`, `val`, and `cal` ranges.
  - Inserts embargo gaps of length $\ge L$ (window length, default 128) between partitions so adjacent windows cannot share raw observations.
  - Fits `IndustrialScaler` **strictly on the allocated `train` slice**.
  - Disallows duplicate window assignment at tiny budgets ($N_{\text{budget}} = 1$).
- **Disjoint Partition Assertions**: Enforced pairwise disjoint window sets in `allocate_target_budget`.

### Task 3: Score and Threshold Scale Alignment
- **Reconstruction Semantics Match**: Updated [`AnomalyEvaluator.evaluate`](file:///F:/ORBIT_Dataset/src/orbit/evaluation/evaluator.py#L103-L150):
  - Reconstructs continuous timestamp scores on the calibration split using the exact same aggregation (`mean` or `max`) as the test evaluation.
  - Calibrates threshold strictly on the timestamp error distribution.
  - Supports external source threshold passing via `external_threshold`.

### Task 4: One-to-One Bipartite Event Matching
- **Strict Bipartite Matching**: Rewrote [`compute_event_metrics`](file:///F:/ORBIT_Dataset/src/orbit/evaluation/metrics.py#L204-L280):
  - Replaces many-to-one detection with greedy maximum-overlap one-to-one matching.
  - Each ground truth event can match at most one predicted alarm.
  - Each predicted alarm episode can match at most one ground truth event.
  - Prevents precision/recall inflation from fragmented alarms.

### Task 5: Valid, Non-Crashing 0% Zero-Shot Protocol
- **Standalone Zero-Shot Handler**: Updated [`experiments/run_transfer_experiments.py`](file:///F:/ORBIT_Dataset/experiments/run_transfer_experiments.py#L135-L185):
  - Budget 0% executes an inference-only path.
  - No training loops (`trainer.fit`) or untrained adapter/scratch models are instantiated.
  - Safely skips schema-mismatched zero-shot when unmapped, with transparent documentation.
  - Passes external threshold without requiring a calibration loader.

### Task 6: Test Portability & Config Safety
- **Portable Tests**: Updated [`tests/test_config.py`](file:///F:/ORBIT_Dataset/tests/test_config.py) to use mock project directories and skippable data integration tests.
- **IndustrialScaler numpy Support**: Extended [`IndustrialScaler`](file:///F:/ORBIT_Dataset/src/orbit/data/preprocessing.py#L110-L150) to support direct fitting and transforming on `np.ndarray` without requiring DataFrame wrappers.

### Task 7: Fair Baselines & Multi-Seed Alignment
- **Added LSTM Baseline**: Integrated `LSTMAutoencoder` into [`experiments/run_transfer_experiments.py`](file:///F:/ORBIT_Dataset/experiments/run_transfer_experiments.py#L210) so LSTM, Scratch Transformer, Adapter, and Fine-Tune run on identical splits and budget constraints.
- **Executed Real-Data Smoke Run**: Executed 1% transfer experiment on real SWaT data:
  - LSTM: AUROC = 0.5715, F1 = 0.0471, Event F1 = 0.0800
  - Scratch Transformer: AUROC = 0.5581, F1 = 0.0478, Event F1 = 0.0408
  - Pretrained + Adapter: AUROC = 0.5483, F1 = 0.0511, Event F1 = 0.1081

---

## 3. Test Suite Status

Total tests collected and passed: **48 passed in 3.54s**.
- `tests/test_adaptation.py`: 10 passed
- `tests/test_config.py`: 7 passed
- `tests/test_data_budgets.py`: 6 passed
- `tests/test_data_pipeline.py`: 6 passed
- `tests/test_evaluation_timestamps.py`: 6 passed
- `tests/test_lstm_baseline.py`: 4 passed
- `tests/test_masked_pretraining.py`: 4 passed
- `tests/test_smoke_transfer.py`: 1 passed
- `tests/test_transformer_baseline.py`: 4 passed

