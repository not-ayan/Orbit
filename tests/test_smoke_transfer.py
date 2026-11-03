"""
End-to-end smoke test for the complete ORBIT transfer learning workflow.
Validates end-to-end execution without requiring multi-gigabyte raw datasets.
"""

from pathlib import Path
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from orbit.data.windowing import SlidingWindowDataset
from orbit.adaptation.transfer import allocate_target_budget, BudgetProtocol
from orbit.adaptation.adapter import AdaptedTransformerModel, AdaptationMode
from orbit.training.trainer import ModelTrainer
from orbit.evaluation.evaluator import AnomalyEvaluator
from orbit.models.temporal_encoder import TransformerAutoencoder


def test_synthetic_end_to_end_transfer_smoke(tmp_path):
    # 1. Create synthetic target normal training and test datasets
    # 500 normal timesteps, 16 features
    train_data = np.sin(np.linspace(0, 50, 500))[:, None].repeat(16, axis=1).astype(np.float32)
    # 300 test timesteps with an anomaly event at 150..180
    test_data = np.sin(np.linspace(0, 30, 300))[:, None].repeat(16, axis=1).astype(np.float32)
    test_labels = np.zeros(300, dtype=np.int64)
    test_data[150:181] += 3.0  # Anomaly burst
    test_labels[150:181] = 1

    train_ds = SlidingWindowDataset(train_data, window_size=32, stride=8)
    test_ds = SlidingWindowDataset(test_data, labels=test_labels, window_size=32, stride=8)

    # 2. Allocate an honest 10% data budget
    alloc = allocate_target_budget(train_ds, budget_percent=10.0, protocol=BudgetProtocol.STRICT_INCLUSIVE)
    meta = alloc.metadata
    assert meta["num_train_windows"] >= 1
    assert meta["num_cal_windows"] >= 1

    train_loader = DataLoader(alloc.train_subset, batch_size=4, shuffle=True)
    cal_loader = DataLoader(alloc.cal_subset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # 3. Initialize Adapted Transformer Model
    model = AdaptedTransformerModel(
        target_dim=16,
        d_model=32,
        num_layers=2,
        d_adapter=8,
        mode=AdaptationMode.ADAPTER,
    )
    param_summary = model.get_parameter_summary()
    assert param_summary["trainable_percent"] < 40.0

    # 4. Train with ModelTrainer
    trainer = ModelTrainer(
        model=model,
        learning_rate=1e-3,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    trainer.fit(train_loader, epochs=2)
    model.mark_calibrated()

    # 5. Evaluate with timestamp reconstruction and event metrics
    evaluator = AnomalyEvaluator(model, aggregation="mean", matching_rule="overlap")
    metrics = evaluator.evaluate(
        val_loader=cal_loader,
        test_loader=test_loader,
        threshold_method="percentile",
        percentile=95.0,
        output_dir=tmp_path / "results",
    )

    # Verify metrics properties
    assert 0.0 <= metrics.timestamp_auroc <= 1.0
    assert 0.0 <= metrics.timestamp_f1 <= 1.0
    assert metrics.num_gt_events == 1
    assert metrics.covered_timesteps > 0
    assert (tmp_path / "results" / "metrics.json").exists()

