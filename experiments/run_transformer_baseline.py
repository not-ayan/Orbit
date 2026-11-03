"""
Experiment Runner for Temporal Transformer Baseline (Trained from Scratch).
Tracks reproducible training, checkpointing, and zero-leakage validation evaluation.
"""

import argparse
from pathlib import Path
import sys
import json

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from orbit.data.pipeline import TelemetryPipeline
from orbit.models.temporal_encoder import TransformerAutoencoder
from orbit.training.trainer import ModelTrainer, set_seed
from orbit.evaluation.evaluator import AnomalyEvaluator


def run_experiment():
    parser = argparse.ArgumentParser(description="ORBIT Transformer Baseline (Trained from Scratch)")
    parser.add_argument("--dataset", type=str, default="swat", choices=["swat", "wadi"])
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer latent dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of Transformer encoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Feedforward network dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--threshold-percentile", type=float, default=99.0, help="Validation threshold percentile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Subset for rapid development")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Subset for rapid development")
    args = parser.parse_args()

    set_seed(args.seed)
    ds = args.dataset.lower()
    print(f"\n{'='*30} ORBIT TRANSFORMER BASELINE: {ds.upper()} {'='*30}")

    if ds == "swat":
        data_dir = Path("datasets/SWaT.A8_June 2021/For Mark")
    else:
        data_dir = Path("datasets/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019")

    out_dir = Path(f"results/transformer_baseline/{ds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline
    pipeline = TelemetryPipeline(
        dataset_name=ds,
        data_dir=data_dir,
        window_size=128,
        stride_train=32,
        stride_test=32,
        val_ratio=0.2,
        scale_method="standard",
        drop_constants=True,
        batch_size=args.batch_size,
    )
    train_ds, val_ds, test_ds = pipeline.run(
        normal_max_rows=args.max_train_rows,
        attack_max_rows=args.max_test_rows,
    )
    train_loader, val_loader, test_loader = pipeline.get_dataloaders(train_ds, val_ds, test_ds)

    input_dim = len(pipeline.sensor_columns)
    print(f"\n[Model Config] Input dim: {input_dim} | d_model: {args.d_model} | Heads: {args.nhead} | Layers: {args.num_layers} | FFN: {args.dim_feedforward}")

    # 2. Model Initialization
    model = TransformerAutoencoder(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )

    # Save model hyperparameter config
    config_dict = {
        "dataset": ds,
        "input_dim": input_dim,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "threshold_percentile": args.threshold_percentile,
    }
    with open(out_dir / "experiment_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 3. Training
    trainer = ModelTrainer(
        model=model,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=out_dir,
    )
    trainer.fit(train_loader, val_loader, epochs=args.epochs)

    # 4. Evaluation
    evaluator = AnomalyEvaluator(model, aggregation="mean")
    metrics = evaluator.evaluate(
        val_loader=val_loader,
        test_loader=test_loader,
        threshold_method="percentile",
        percentile=args.threshold_percentile,
        output_dir=out_dir,
    )

    print(f"\n[SUMMARY] Experiment finished for {ds.upper()}. Checkpoint & metrics saved to {out_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_experiment()

