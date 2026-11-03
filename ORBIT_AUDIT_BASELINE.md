# ORBIT Audit Baseline Snapshot

**Generated Date**: 2026-11-03  
**Working Directory**: `F:\ORBIT_Dataset`  
**Execution Pack**: `F:\ORBIT_Dataset\ORBIT_Agent_Execution_Pack`

---

## 1. Environment & Hardware Inventory

- **Operating System**: Windows (AMD64)
- **Python Version**: `3.14.0b1 (tags/v3.14.0b1:b092705, May 7 2025, 10:22:31) [MSC v.1943 64 bit (AMD64)]`
- **PyTorch Version**: `2.15.0+cpu` (CUDA available: `False`, Hardware: `CPU`)
- **NumPy Version**: `2.5.4`
- **scikit-learn Version**: `1.9.1`
- **Pandas Version**: `3.1.0`
- **pytest Version**: `9.2.0`
- **Git Status**: Workspace is not currently initialized as a git repository (`fatal: not a git repository`). No files modified or deleted during audit.

---

## 2. Dataset Accessibility & Integrity

1. **SWaT (Secure Water Treatment) A8 (June 2021)**:
   - Location: `datasets/SWaT.A8_June 2021/For Mark`
   - Content: 36 raw CSV telemetry files partitioned across dates 2021-06-22 to 2021-07-21.
   - Status: Accessible and verified.
2. **WADI (Water Distribution) A2 (Nov 2019)**:
   - Location: `datasets/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019`
   - Content: `WADI_14days_new.csv` (normal baseline) and `WADI_attackdataLABLE.csv` (attack telemetry).
   - Status: Accessible and verified.
3. **Processed Artifacts**:
   - Location: `data/processed/swat/` and `data/processed/wadi/`
   - Files: `pipeline_metadata.json`, `scaler.json`
   - Status: Accessible and verified.
4. **Feasibility of Rerunning Real-Data Experiments**:
   - Real-data experiments can be run directly from raw files using `TelemetryPipeline`.
   - Hardware is CPU-only, so full fine-tuning over thousands of windows is compute-bound. Synthetic test suites and focused real-data calibration runs will be utilized for rapid regression testing.

---

## 3. Existing Checkpoints & Artifacts

- Pretrained Backbone (WADI): `results/pretraining/wadi/pretrained_backbone.pt` (745 KB)
- Full Pretrained Model (WADI): `results/pretraining/wadi/best_pretrain_model.pt` (745 KB)
- Transformer Baseline Checkpoints:
  - SWaT: `results/transformer_baseline/swat/best_model.pt` (773 KB)
  - WADI: `results/transformer_baseline/wadi/best_model.pt` (878 KB)
- LSTM Baseline Checkpoints:
  - SWaT: `results/lstm_baseline/swat/best_model.pt` (124 KB)
  - WADI: `results/lstm_baseline/wadi/best_model.pt` (186 KB)
- Few-Shot Adaptation Artifacts:
  - `results/transfer_experiments/wadi_to_swat/benchmark_summary.json`
  - Per-budget checkpoint directories under `results/transfer_experiments/wadi_to_swat/`

All existing checkpoints are preserved untouched.

---

## 4. Verification of Known Issues

| Issue | Execution Pack Hypothesis | Live Code Status | Evidence / Affected Code |
| :--- | :--- | :--- | :--- |
| **A. Zero-Shot Path & Checkpoint Completeness** | Zero-shot path loads only backbone; target projection and reconstruction head are randomly initialized and frozen. Target-specific projection cannot function without target data or schema mapping. | **VERIFIED** | In [`src/orbit/adaptation/adapter.py:AdaptedTransformerModel`](file:///F:/ORBIT_Dataset/src/orbit/adaptation/adapter.py#L101-L171), `target_projection` and `target_recon_head` are initialized with standard PyTorch random weights and frozen (`requires_grad = False`) under `AdaptationMode.ZERO_SHOT`. Furthermore, `load_pretrained_backbone()` only loads `transformer_encoder` and `pos_encoder`, leaving projections unmapped. |
| **B. Data Budget Accounting** | Data budgets (0%, 1%, 5%, 10%) count only training windows while using the full target validation split (e.g. 528 windows) for early stopping and threshold calibration. | **VERIFIED** | In [`experiments/run_transfer_experiments.py`](file:///F:/ORBIT_Dataset/experiments/run_transfer_experiments.py#L74-L167), 1% budget allocates 21 training windows via `partition_target_budget`, but `trainer.fit(budget_loader, val_loader)` and `evaluator.evaluate(val_loader=val_loader)` consume the entire 528-window validation set ($>25\times$ the adaptation budget). |
| **C. Timestamp-Level Scoring & Event Metrics** | Window-level reconstruction scores and window-level labels (`ANY` positive policy) are evaluated and reported as point-wise F1 and AUROC without continuous timestamp projection or proper event matching. | **VERIFIED** | In [`src/orbit/evaluation/evaluator.py`](file:///F:/ORBIT_Dataset/src/orbit/evaluation/evaluator.py#L58-L76), `per_step_err` is averaged across time to produce a 1D scalar per window `window_scores = per_step_err.mean(dim=-1)`, and labels are reduced via `(batch_y.sum(dim=-1) > 0)`. The metric output in [`src/orbit/evaluation/metrics.py`](file:///F:/ORBIT_Dataset/src/orbit/evaluation/metrics.py#L123-L140) names this `point_f1` despite being window-level. |
| **D. Config & Checkpoint Safety** | Hardcoded dataset paths in transfer scripts; fragile checkpoint loading without schema/domain identity verification. | **VERIFIED** | In [`experiments/run_transfer_experiments.py`](file:///F:/ORBIT_Dataset/experiments/run_transfer_experiments.py#L51), `target_data_dir = Path("datasets/SWaT.A8_June 2021/For Mark")` is hardcoded. Pretrained backbone loader does not verify metadata compatibility with source dataset. |
| **E. Fair Baselines** | Baselines (scratch vs adapter vs fine-tuning) do not have identical split protocols and target-data access accounting. | **VERIFIED** | Scratch Transformer and Adapter models used different target data access assumptions (e.g., full validation split vs budgeted validation). |
| **F. Multi-Seed Reporting** | Benchmark reports results from a single random seed (`seed=42`) without standard deviation or replication across seeds. | **VERIFIED** | In [`experiments/run_transfer_experiments.py`](file:///F:/ORBIT_Dataset/experiments/run_transfer_experiments.py#L35), `seed` defaults to 42 and no multi-seed aggregation is implemented. |
| **G. Packaging & Environment** | Tests require manual `$env:PYTHONPATH = "src"`. No `pyproject.toml` or editable install exists. | **VERIFIED** | Running `.venv\Scripts\pytest.exe tests/ -q` without `PYTHONPATH` fails with 5 collection errors (`ModuleNotFoundError: No module named 'orbit'`). Neither `pyproject.toml` nor `setup.py` exists. |

---

## 5. Baseline Test Execution

Command executed:
```powershell
$env:PYTHONPATH = "src"; .venv\Scripts\pytest.exe tests/ -v
```
Result:
```text
============================= test session starts =============================
platform win32 -- Python 3.14.0b1, pytest-9.2.0, pluggy-1.7.0 -- F:\ORBIT_Dataset\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: F:\ORBIT_Dataset
collecting ... collected 22 items

tests/test_adaptation.py::test_temporal_adapter_layer PASSED             [  4%]
tests/test_adaptation.py::test_adapted_transformer_modes_and_freezing PASSED [  9%]
tests/test_adaptation.py::test_load_real_pretrained_backbone PASSED      [ 13%]
tests/test_adaptation.py::test_partition_target_budget PASSED            [ 18%]
tests/test_data_pipeline.py::test_chronological_split PASSED             [ 22%]
tests/test_data_pipeline.py::test_missing_value_imputation PASSED        [ 27%]
tests/test_data_pipeline.py::test_constant_column_detection PASSED       [ 31%]
tests/test_data_pipeline.py::test_industrial_scaler_zero_leakage PASSED  [ 36%]
tests/test_data_pipeline.py::test_sliding_window_dataset_shapes_and_policies PASSED [ 40%]
tests/test_data_pipeline.py::test_numpy_sliding_window_vectorized PASSED [ 45%]
tests/test_lstm_baseline.py::test_lstm_autoencoder_forward_shape PASSED  [ 50%]
tests/test_lstm_baseline.py::test_reconstruction_error_reductions PASSED [ 54%]
tests/test_lstm_baseline.py::test_threshold_selection_and_event_metrics PASSED [ 59%]
tests/test_lstm_baseline.py::test_trainer_and_evaluator_integration PASSED [ 63%]
tests/test_masked_pretraining.py::test_mask_generator_strategies PASSED  [ 68%]
tests/test_masked_pretraining.py::test_orbit_pretrainer_masked_loss_computation PASSED [ 72%]
tests/test_masked_pretraining.py::test_orbit_backbone_export PASSED      [ 77%]
tests/test_masked_pretraining.py::test_masked_pretrainer_integration PASSED [ 81%]
tests/test_transformer_baseline.py::test_domain_projections PASSED       [ 86%]
tests/test_transformer_baseline.py::test_positional_encoding PASSED      [ 90%]
tests/test_transformer_baseline.py::test_transformer_autoencoder_forward PASSED [ 95%]
tests/test_transformer_baseline.py::test_transformer_trainer_and_evaluator_integration PASSED [100%]

============================= 22 passed in 3.45s ==============================
```

---

## 6. Audit Conclusion & Next Task

All 7 known issues have been verified against the live codebase.
No user files or checkpoints were modified or deleted during this audit.
Task 0 is complete. Proceeding immediately to **Task 1: Repair zero-shot inference and checkpoint completeness**.

