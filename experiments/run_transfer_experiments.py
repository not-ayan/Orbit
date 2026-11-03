"""
Comprehensive Cross-Domain Transfer & Few-Shot Adaptation Benchmark for ORBIT.

Tests the Central Research Question of ORBIT:
"Can self-supervised pretraining improve anomaly detection when a multivariate
time-series model adapts to a previously unseen industrial system using only a
small amount of target-domain data?"

Protocol Guarantees:
1. Multi-Seed Replication: Computes mean +/- std across independent seeds.
2. Honest Budget Accounting: The declared budget (e.g. 1%, 5%, 10%) strictly covers
   all target data accessed (adaptation + validation + calibration).
3. Fair Baseline Comparison: Evaluates LSTM Autoencoder, Scratch Transformer,
   Pretrained + Adapter, and Pretrained Full Fine-Tuning under identical splits.
4. Separate Zero-Shot Protocol: 0% target data does not instantiate untrained adapters
   or call training loops.
5. Timestamp-Level Resolution: Reconstructs continuous 1D timeline scores from sliding windows.
6. One-to-One Event Matching: Enforces strict one-to-one interval matching without double counting.
7. Zero Test Leakage: Threshold is calibrated strictly on target calibration data; test labels
   are accessed only once during final scoring.
"""

import argparse
from pathlib import Path
import sys
import json
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from orbit.config import (
    resolve_dataset_dir,
    validate_checkpoint_for_transfer,
    load_yaml_config,
)
from orbit.data.pipeline import TelemetryPipeline
from orbit.models.lstm_autoencoder import LSTMAutoencoder
from orbit.models.temporal_encoder import TransformerAutoencoder
from orbit.models.checkpoint import CheckpointValidationError
from orbit.adaptation.adapter import AdaptationMode, AdaptedTransformerModel
from orbit.adaptation.transfer import (
    BudgetProtocol,
    allocate_target_budget,
)
from orbit.training.trainer import ModelTrainer, set_seed
from orbit.evaluation.evaluator import AnomalyEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="ORBIT Few-Shot Adaptation Benchmark")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration")
    parser.add_argument("--source-dataset", type=str, default="wadi")
    parser.add_argument("--target-dataset", type=str, default="swat")
    parser.add_argument("--data-dir", type=str, default=None, help="Explicit target dataset directory")
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.0, 1.0, 5.0, 10.0, 100.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=5, help="Adaptation epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adaptation learning rate")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-adapter", type=int, default=16)
    parser.add_argument("--backbone-path", type=str, default="results/pretraining/wadi/pretrained_backbone.pt")
    parser.add_argument("--out-dir", type=str, default=None)
    return parser.parse_args()


def run_benchmark():
    args = parse_args()

    # Load YAML configuration if supplied
    if args.config:
        cfg = load_yaml_config(args.config)
        exp_cfg = cfg.get("experiment", {})
        args.source_dataset = exp_cfg.get("source_dataset", args.source_dataset)
        args.target_dataset = exp_cfg.get("target_dataset", args.target_dataset)
        args.seeds = exp_cfg.get("seeds", args.seeds)
        args.budgets = exp_cfg.get("budgets", args.budgets)
        train_cfg = cfg.get("training", {})
        args.epochs = train_cfg.get("epochs", args.epochs)
        args.batch_size = train_cfg.get("batch_size", args.batch_size)
        args.lr = train_cfg.get("lr", args.lr)
        paths_cfg = cfg.get("paths", {})
        args.backbone_path = paths_cfg.get("backbone_checkpoint", args.backbone_path)

    print(f"\n{'='*25} ORBIT TRANSFER BENCHMARK {'='*25}")
    print(f"Source: {args.source_dataset.upper()} -> Target: {args.target_dataset.upper()}")
    print(f"Seeds: {args.seeds}")
    print(f"Budgets: {args.budgets}%")
    print(f"Backbone: {args.backbone_path}")

    # Validate dataset directory and backbone checkpoint
    target_data_dir = resolve_dataset_dir(args.target_dataset, custom_dir=args.data_dir)
    validate_checkpoint_for_transfer(
        checkpoint_path=args.backbone_path,
        expected_d_model=args.d_model,
    )

    out_root = Path(args.out_dir or f"results/transfer_experiments/{args.source_dataset}_to_{args.target_dataset}")
    out_root.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Target Data Pipeline
    pipeline = TelemetryPipeline(
        dataset_name=args.target_dataset,
        data_dir=target_data_dir,
        window_size=128,
        stride_train=32,
        stride_test=32,
        val_ratio=0.2,
        scale_method="standard",
        drop_constants=True,
        batch_size=args.batch_size,
    )
    full_train_ds, full_val_ds, test_ds = pipeline.run()
    _, _, test_loader = pipeline.get_dataloaders(full_train_ds, full_val_ds, test_ds)
    target_dim = len(pipeline.sensor_columns)

    all_seed_runs: List[Dict[str, Any]] = []

    # 2. Iterate across random seeds
    for seed in args.seeds:
        set_seed(seed)
        seed_dir = out_root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'-'*65}")
        print(f">>> EXECUTING BENCHMARK RUN: SEED {seed}")

        for budget in args.budgets:
            print(f"\n  [Seed {seed}] Evaluating Budget: {budget}%")

            # Dedicated 0% Strict Zero-Shot Protocol (Task 5)
            if budget == 0.0:
                print("    Executing Strict Zero-Shot Protocol (0% target data)...")
                # Strict zero-shot check
                try:
                    zs_model = AdaptedTransformerModel(
                        target_dim=target_dim,
                        pretrained_backbone_path=args.backbone_path,
                        d_model=args.d_model,
                        mode=AdaptationMode.ZERO_SHOT,
                    )
                    zs_model.validate_for_inference()
                    # If valid, evaluate with external source threshold
                    evaluator = AnomalyEvaluator(zs_model, aggregation="mean", matching_rule="overlap")
                    metrics = evaluator.evaluate(
                        val_loader=None,
                        test_loader=test_loader,
                        external_threshold=1.0,
                    )
                    record = {
                        "seed": seed,
                        "name": "Strict Zero-Shot (0%)",
                        "type": "strict_zero_shot",
                        "budget_percent": 0.0,
                        "num_budgeted_windows": 0,
                        "num_train_windows": 0,
                        "num_cal_windows": 0,
                        "trainable_params": 0,
                        "trainable_percent": 0.0,
                        "timestamp_auroc": metrics.timestamp_auroc,
                        "timestamp_pr_auc": metrics.timestamp_pr_auc,
                        "timestamp_f1": metrics.timestamp_f1,
                        "timestamp_precision": metrics.timestamp_precision,
                        "timestamp_recall": metrics.timestamp_recall,
                        "event_f1": metrics.event_f1,
                        "event_precision": metrics.event_precision,
                        "event_recall": metrics.event_recall,
                        "num_detected_events": metrics.num_detected_events,
                        "num_gt_events": metrics.num_gt_events,
                        "covered_timesteps": metrics.covered_timesteps,
                    }
                    all_seed_runs.append(record)
                except CheckpointValidationError as e:
                    print(f"    [Zero-Shot Info] Mismatched schemas without sensor mapping cannot run strict zero-shot: {e}")
                    # Document transparently without crashing
                    continue
                continue

            # Few-shot budgets (> 0.0%)
            alloc = allocate_target_budget(
                full_train_ds,
                budget_percent=budget,
                protocol=BudgetProtocol.STRICT_INCLUSIVE,
            )
            meta = alloc.metadata
            print(f"    Allocated Windows -> Train: {meta['num_train_windows']}, "
                  f"Val: {meta['num_val_windows']}, Cal: {meta['num_cal_windows']} "
                  f"(Total Budgeted: {meta['total_budgeted_windows']} / {len(full_train_ds)})")

            # Create data loaders for budgeted splits
            train_loader = (
                DataLoader(alloc.train_subset, batch_size=min(args.batch_size, len(alloc.train_subset)), shuffle=True)
                if alloc.train_subset and len(alloc.train_subset) > 0
                else None
            )
            val_loader = (
                DataLoader(alloc.val_subset, batch_size=min(args.batch_size, len(alloc.val_subset)), shuffle=False)
                if alloc.val_subset and len(alloc.val_subset) > 0
                else None
            )
            cal_loader = (
                DataLoader(alloc.cal_subset, batch_size=min(args.batch_size, len(alloc.cal_subset)), shuffle=False)
                if alloc.cal_subset and len(alloc.cal_subset) > 0
                else train_loader
            )

            # Define models to benchmark under this budget (Task 5 & 7)
            configs_to_test = []

            # 1. Baseline 1: LSTM Autoencoder (Control)
            configs_to_test.append({
                "name": f"LSTM Autoencoder ({budget}%)",
                "type": "lstm_autoencoder",
                "mode": None,
            })
            # 2. Baseline 2: Scratch Transformer (Control)
            configs_to_test.append({
                "name": f"Scratch Transformer ({budget}%)",
                "type": "scratch_transformer",
                "mode": None,
            })
            # 3. Core Method: Pretrained + Adapter
            configs_to_test.append({
                "name": f"Pretrained + Adapter ({budget}%)",
                "type": "pretrained_adapter",
                "mode": AdaptationMode.ADAPTER,
            })
            # 4. Optional: Pretrained + Full Fine-Tune (for budgets >= 5%)
            if budget in [5.0, 10.0, 100.0]:
                configs_to_test.append({
                    "name": f"Pretrained + Full Fine-Tune ({budget}%)",
                    "type": "pretrained_finetune",
                    "mode": AdaptationMode.FULL_FINETUNE,
                })

            for cfg in configs_to_test:
                run_name = cfg["name"]
                run_out_dir = seed_dir / f"budget_{int(budget)}pct_{cfg['type']}"
                run_out_dir.mkdir(parents=True, exist_ok=True)

                if cfg["type"] == "lstm_autoencoder":
                    model = LSTMAutoencoder(
                        input_dim=target_dim,
                        hidden_dim=64,
                        latent_dim=32,
                        num_layers=2,
                        dropout=0.1,
                    )
                    param_summary = {
                        "total_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_percent": 100.0,
                    }
                elif cfg["type"] == "scratch_transformer":
                    model = TransformerAutoencoder(
                        input_dim=target_dim,
                        d_model=args.d_model,
                        nhead=4,
                        num_layers=3,
                        dim_feedforward=256,
                        dropout=0.1,
                    )
                    param_summary = {
                        "total_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_percent": 100.0,
                    }
                else:
                    model = AdaptedTransformerModel(
                        target_dim=target_dim,
                        pretrained_backbone_path=args.backbone_path,
                        d_model=args.d_model,
                        d_adapter=args.d_adapter,
                        mode=cfg["mode"],
                    )
                    param_summary = model.get_parameter_summary()

                # Train model on budgeted target data
                if train_loader is not None and param_summary["trainable_parameters"] > 0:
                    trainer = ModelTrainer(
                        model=model,
                        learning_rate=args.lr,
                        patience=2,
                        checkpoint_dir=run_out_dir,
                    )
                    trainer.fit(train_loader, val_loader=val_loader, epochs=args.epochs)
                    if hasattr(model, "mark_calibrated"):
                        model.mark_calibrated()

                # Evaluate with timestamp-level resolution and one-to-one event metrics
                evaluator = AnomalyEvaluator(model, aggregation="mean", matching_rule="overlap")
                metrics = evaluator.evaluate(
                    val_loader=cal_loader,
                    test_loader=test_loader,
                    threshold_method="percentile",
                    percentile=99.0,
                    output_dir=run_out_dir,
                )

                record = {
                    "seed": seed,
                    "name": run_name,
                    "type": cfg["type"],
                    "budget_percent": budget,
                    "num_budgeted_windows": meta["total_budgeted_windows"],
                    "num_train_windows": meta["num_train_windows"],
                    "num_cal_windows": meta["num_cal_windows"],
                    "trainable_params": param_summary["trainable_parameters"],
                    "trainable_percent": param_summary["trainable_percent"],
                    "timestamp_auroc": metrics.timestamp_auroc,
                    "timestamp_pr_auc": metrics.timestamp_pr_auc,
                    "timestamp_f1": metrics.timestamp_f1,
                    "timestamp_precision": metrics.timestamp_precision,
                    "timestamp_recall": metrics.timestamp_recall,
                    "event_f1": metrics.event_f1,
                    "event_precision": metrics.event_precision,
                    "event_recall": metrics.event_recall,
                    "num_detected_events": metrics.num_detected_events,
                    "num_gt_events": metrics.num_gt_events,
                    "covered_timesteps": metrics.covered_timesteps,
                }
                all_seed_runs.append(record)

    # 3. Compute Aggregate Statistics across seeds
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_seed_runs:
        key = f"{r['type']}_{int(r['budget_percent'])}pct"
        grouped.setdefault(key, []).append(r)

    summary_table: List[Dict[str, Any]] = []
    print(f"\n{'='*75}")
    print(f"{'EXPERIMENT CONFIGURATION':<34} | {'AUROC (mean±std)':<18} | {'F1 (mean±std)':<18}")
    print(f"{'-'*75}")

    for key, runs in grouped.items():
        sample = runs[0]
        aurocs = [r["timestamp_auroc"] for r in runs]
        f1s = [r["timestamp_f1"] for r in runs]
        pr_aucs = [r["timestamp_pr_auc"] for r in runs]
        ev_f1s = [r["event_f1"] for r in runs]

        summary_entry = {
            "name": sample["name"],
            "type": sample["type"],
            "budget_percent": sample["budget_percent"],
            "num_seeds": len(runs),
            "num_train_windows": sample["num_train_windows"],
            "auroc_mean": float(np.mean(aurocs)),
            "auroc_std": float(np.std(aurocs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "pr_auc_mean": float(np.mean(pr_aucs)),
            "pr_auc_std": float(np.std(pr_aucs)),
            "event_f1_mean": float(np.mean(ev_f1s)),
            "event_f1_std": float(np.std(ev_f1s)),
            "per_seed_runs": runs,
        }
        summary_table.append(summary_entry)

        auroc_str = f"{np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}"
        f1_str = f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        print(f"{sample['name']:<34} | {auroc_str:<18} | {f1_str:<18}")

    print(f"{'='*75}")

    # Save complete multi-seed benchmark summary
    summary_path = out_root / "multi_seed_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)
    print(f"\nMulti-seed summary saved to: {summary_path}")


if __name__ == "__main__":
    run_benchmark()
