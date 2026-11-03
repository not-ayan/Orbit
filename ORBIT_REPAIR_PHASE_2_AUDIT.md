# ORBIT Repair Phase 2 — Audit & Issue Verification

**Generated Date**: 2026-11-03  
**Working Directory**: `F:\ORBIT_Dataset`  
**Git Head**: `bfd6a08` (`feat(orbit): comprehensive research code repair and validation (Tasks 0-6)`)

---

## 1. Environment & Hardware Inventory

- **Python Version**: `3.14.0b1`
- **PyTorch Version**: `2.15.0+cpu` (CPU only)
- **NumPy Version**: `2.5.4`
- **scikit-learn Version**: `1.9.1`
- **Pandas Version**: `3.1.0`
- **PyYAML Version**: `6.0.3`
- **pytest Version**: `9.2.0`
- **Raw Datasets Available**: SWaT A8 (36 CSVs) and WADI A2 (2 CSVs).
- **Existing Tests**: 40 tests collected, 40 passed in 3.57s.

---

## 2. Verification Matrix of Review Findings A–G

| Issue ID | Defect Hypothesis | Live Code Status | Affected File & Symbol | Evidence / Reproduction Command |
| :--- | :--- | :--- | :--- | :--- |
| **A.1** | Tensor boolean ambiguity (`tensor or None`) causing `RuntimeError`. | **VERIFIED** | [`src/orbit/adaptation/adapter.py`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/adapter.py#L225-L226): `source_in_w = state.get(...) or state.get(...)`. | In PyTorch, evaluating `tensor or None` on a multi-element tensor raises `RuntimeError: Boolean value of Tensor with more than one value is ambiguous`. |
| **A.2 & A.3** | Encoder-only backbone lacks projection/head; uncalibrated zero-shot risks using random frozen layers. | **VERIFIED** | [`src/orbit/adaptation/adapter.py`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/adapter.py#L180-L245): `load_pretrained_checkpoint()` and `AdaptedTransformerModel`. | An encoder-only checkpoint only loads encoder weights. If $F_{\text{target}} \ne F_{\text{source}}$, target projections are random unless properly blocked or calibrated. |
| **A.4** | Sensor mapping metadata stored but not applied in `forward()`. | **VERIFIED** | [`src/orbit/adaptation/adapter.py:AdaptedTransformerModel.forward`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/adapter.py#L321-L342). | `self.sensor_mapping` is accepted in `__init__`, but `forward(x)` directly computes `self.target_projection(x)` without applying the mapping! |
| **A.5** | Evaluator calls `model(batch_x)` directly, bypassing inference validation. | **VERIFIED** | [`src/orbit/evaluation/evaluator.py:compute_step_errors`](file:///F:/ORBIT_Dataset/src/orbit/evaluation/evaluator.py#L52). | Evaluator runs `recon_x = self.model(batch_x)`, which completely bypasses `validate_for_inference()` in `compute_reconstruction_error`. |
| **B.1** | Scaler and constant-feature removal fitted on full target training set before budget allocation. | **VERIFIED** | [`src/orbit/data/pipeline.py:TelemetryPipeline.run`](file:///F:/ORBIT_Dataset/src/orbit/data/pipeline.py#L90-L125). | The pipeline fits `IndustrialScaler` on the entire 80% train split before `run_transfer_experiments.py` allocates the 1% or 5% budget. |
| **B.2 & B.3** | Train, validation, and calibration window subsets overlap in original raw timestamps. | **VERIFIED** | [`src/orbit/adaptation/transfer.py:allocate_target_budget`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/transfer.py#L95-L120). | With window length 128 and stride 32, window 14 (samples 448..576) overlaps window 15 (samples 480..608). Consecutive window indices share raw observations across split boundaries. |
| **C** | Threshold calibrated on window-average scores but applied to timestamp-reconstructed scores. | **VERIFIED** | [`src/orbit/evaluation/evaluator.py:evaluate`](file:///F:/ORBIT_Dataset/src/orbit/evaluation/evaluator.py#L97-L120). | `threshold = select_validation_threshold(val_window_scores)` computes threshold on window-mean MSE, but evaluates against `timestamp_scores`. Different distributions and variances. |
| **D** | Event metrics allow many-to-one matching, inflating event precision/recall. | **VERIFIED** | [`src/orbit/evaluation/metrics.py:compute_event_metrics`](file:///F:/ORBIT_Dataset/src/orbit/evaluation/metrics.py#L225-L270). | Multiple fragmented predicted alarms hitting the same ground-truth event are all counted as true positives ($valid\_alarms$). Lacks one-to-one bipartite matching. |
| **E** | 0% budget passes `None` calibration loader to evaluator, crashing with `TypeError`. | **VERIFIED** | [`experiments/run_transfer_experiments.py`](file:///F:/ORBIT_Dataset/experiments/run_transfer_experiments.py#L163). | At budget 0%, `cal_loader` is `None`; `compute_step_errors(None)` raises `TypeError: 'NoneType' object is not iterable`. Also instantiates untrained adapters at 0%. |
| **F** | Tests fail in environments without raw SWaT/WADI directories. | **VERIFIED** | [`tests/test_config.py:test_resolve_dataset_dir_known`](file:///F:/ORBIT_Dataset/tests/test_config.py#L20). | Directly asserts existence of `datasets/SWaT.A8_June 2021` on disk. Will fail on clean clones/CI without the multi-gigabyte dataset. |
| **G** | Baselines omit LSTM few-shot comparison; multi-seed lacks per-seed run verification. | **VERIFIED** | [`experiments/run_transfer_experiments.py`](file:///F:/ORBIT_Dataset/experiments/run_transfer_experiments.py). | LSTM Autoencoder was not included in the few-shot benchmark. |

---

## 3. Plan of Execution

Task 0 is completed. Proceeding directly to:
- **Task 1**: Fix tensor boolean logic, complete inference checkpoints, forward-path sensor mapping, and mandatory inference validation in evaluator.
- **Task 2**: Raw-timestamp chronological splitting before windowing, fitting scalers strictly on permitted budget, and embargo gaps to prevent observation leakage.
- **Task 3**: Timestamp-level score reconstruction on calibration data so threshold and evaluation use identical score definitions.
- **Task 4**: One-to-one maximum cardinality / overlap bipartite event matching.
- **Task 5**: Standalone, non-crashing 0% / strict zero-shot evaluation protocol.
- **Task 6**: Synthetic test fixtures, skippable integration tests, and configuration validation.
- **Task 7**: Unified multi-seed baseline comparisons (LSTM, Scratch Transformer, Adapter, Fine-Tune) and documentation.

