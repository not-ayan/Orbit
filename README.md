# ORBIT: Operational Recognition of Behavioural Intelligence with Transformers

ORBIT investigates self-supervised Transformer-based learning of normal operating behaviour from multivariate industrial time-series telemetry, with the goal of detecting anomalous events when adapting to previously unseen industrial systems using strictly budgeted few-shot target data.

---

## 1. Quickstart & Installation

ORBIT requires Python 3.10+ and PyTorch.

```powershell
# 1. Clone repository and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install editable package with development dependencies
pip install -e .
```

### Run Test Suite
Running the test suite does not require setting `$env:PYTHONPATH`:
```powershell
pytest -v
```
All 48 unit, integration, and regression tests will execute.

---

## 2. Repository Layout

```text
ORBIT_Dataset/
├── configs/                       # Validated YAML experiment configurations
│   ├── data_config.yaml           # Windowing and dataset paths
│   └── transfer_config.yaml       # Multi-seed transfer experiment settings
├── src/orbit/                     # Core library
│   ├── adaptation/                # Residual adapters & target budget accounting
│   │   ├── adapter.py             # TemporalAdapterLayer & AdaptedTransformerModel
│   │   └── transfer.py            # Target data budget partitioner
│   ├── data/                      # Data pipeline & chronological windowing
│   │   ├── loading.py             # SWaT and WADI raw CSV ingestion
│   │   ├── preprocessing.py       # IndustrialScaler & missing value imputation
│   │   ├── windowing.py           # SlidingWindowDataset & vectorized numpy windows
│   │   └── pipeline.py            # High-level TelemetryPipeline
│   ├── evaluation/                # Scoring & evaluation metrics
│   │   ├── metrics.py             # Timestamp reconstruction & event matching
│   │   └── evaluator.py           # AnomalyEvaluator
│   ├── models/                    # Neural architectures & checkpointing
│   │   ├── checkpoint.py          # Versioned checkpoint validation
│   │   ├── temporal_encoder.py    # Transformer autoencoder & projections
│   │   ├── lstm_autoencoder.py    # LSTM Autoencoder baseline
│   │   └── orbit.py               # Masked pretraining model & masking generator
│   └── training/                  # Trainers & early stopping
│       ├── trainer.py             # ModelTrainer
│       └── pretrain_trainer.py    # MaskedPretrainTrainer
├── experiments/                   # Reproducible experiment runners
│   ├── run_lstm_baseline.py       # LSTM Autoencoder baseline runner
│   ├── run_transformer_baseline.py# Scratch Transformer baseline runner
│   ├── run_masked_pretraining.py  # Self-supervised temporal pretraining runner
│   └── run_transfer_experiments.py# Few-shot cross-domain adaptation benchmark
├── tests/                         # Comprehensive pytest test suite (40 tests)
├── RESEARCH_PROTOCOL.md           # Scientific protocol, data budgets, and metrics
├── CHANGELOG_ORBIT_REPAIR.md      # Detailed changelog of repository repairs
└── pyproject.toml                 # Package configuration (PEP 621)
```

---

## 3. Core Scientific Workflows

### 3.1 Data Preparation
Process raw telemetry with train-only normalization (zero test leakage):
```powershell
python -m orbit.data.prepare --dataset wadi
python -m orbit.data.prepare --dataset swat
```

### 3.2 Self-Supervised Pretraining
Pretrain the temporal Transformer backbone on source domain telemetry (WADI) using hybrid block masking:
```powershell
python experiments/run_masked_pretraining.py --dataset wadi --epochs 10 --mask-ratio 0.20
```
This saves the domain-agnostic backbone to `results/pretraining/wadi/pretrained_backbone.pt`.

### 3.3 Few-Shot Adaptation Benchmark
Execute the cross-domain transfer benchmark comparing:
1. **ORBIT Pretrained + Adapter** (frozen backbone, lightweight temporal bottleneck adapter)
2. **Scratch Transformer** (trained only on the budgeted target data)
3. **LSTM Autoencoder Baseline** (recurrent sequence baseline on budgeted target data)
4. **Full Fine-Tuning** (all parameters updated)

```powershell
python experiments/run_transfer_experiments.py --config configs/transfer_config.yaml
```
Optional CLI overrides:
```powershell
python experiments/run_transfer_experiments.py --budgets 1.0 5.0 10.0 --seeds 42 43 44 --epochs 5
```

---

## 4. Evaluation & Scientific Guarantees

- **Timestamp-Level Resolution**: Sliding window reconstruction errors are projected back to continuous 1D physical time via overlapping arithmetic mean aggregation.
- **Honest Target Budgets**: The declared budget (e.g. 1%, 5%, 10%) strictly bounds all target data accessed across adaptation, validation, and threshold calibration.
- **Event-Level Metrics**: Evaluates contiguous operational anomaly episodes using documented interval overlap and IoU matching rules.
- **Zero Test Leakage**: The anomaly threshold is calibrated strictly on normal calibration data; test labels are touched only once during final scoring.

For full scientific definitions, please consult [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md).

