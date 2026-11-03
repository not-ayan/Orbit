"""
Experiment Runner for Self-Supervised Masked Reconstruction Pretraining (ORBIT Step 4).
Pretrains the temporal Transformer backbone on source industrial telemetry.
"""

import argparse
from pathlib import Path
import sys
import json

src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from orbit.data.pipeline import TelemetryPipeline
from orbit.models.orbit import MaskStrategy, ORBITPretrainer
from orbit.training.pretrain_trainer import MaskedPretrainer
from orbit.training.trainer import set_seed


def run_experiment():
    parser = argparse.ArgumentParser(description="ORBIT Self-Supervised Masked Pretraining")
    parser.add_argument("--source-dataset", type=str, default="wadi", choices=["wadi", "swat"])
    parser.add_argument("--mask-ratio", type=float, default=0.20, help="Fraction of tokens/spans to mask")
    parser.add_argument("--mask-strategy", type=str, default="hybrid", choices=["hybrid", "block_time", "random_token"])
    parser.add_argument("--epochs", type=int, default=8, help="Pretraining epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer latent dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Attention heads")
    parser.add_argument("--num-layers", type=int, default=3, help="Transformer encoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-rows", type=int, default=None, help="Subset for rapid development")
    args = parser.parse_args()

    set_seed(args.seed)
    ds = args.source_dataset.lower()
    print(f"\n{'='*25} ORBIT MASKED PRETRAINING: SOURCE = {ds.upper()} {'='*25}")

    if ds == "wadi":
        data_dir = Path("datasets/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019")
    else:
        data_dir = Path("datasets/SWaT.A8_June 2021/For Mark")

    out_dir = Path(f"results/pretraining/{ds}")
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
    )
    train_loader, val_loader, _ = pipeline.get_dataloaders(train_ds, val_ds, test_ds)

    input_dim = len(pipeline.sensor_columns)
    print(f"\n[Pretraining Config] Source: {ds.upper()} | Features: {input_dim} | Mask Ratio: {args.mask_ratio*100:.1f}% | Strategy: {args.mask_strategy}")
    print(f"    Architecture: d_model={args.d_model}, heads={args.nhead}, layers={args.num_layers}, FFN={args.dim_feedforward}")

    # 2. Model Initialization
    model = ORBITPretrainer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        mask_strategy=args.mask_strategy,
        mask_ratio=args.mask_ratio,
    )

    # Save pretraining config
    config_dict = {
        "source_dataset": ds,
        "input_dim": input_dim,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "mask_strategy": args.mask_strategy,
        "mask_ratio": args.mask_ratio,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    with open(out_dir / "pretrain_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # 3. Pretraining
    pretrainer = MaskedPretrainer(
        model=model,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_dir=out_dir,
    )
    pretrainer.fit(train_loader, val_loader, epochs=args.epochs)

    print(f"\n[SUMMARY] Pretraining finished for {ds.upper()}.")
    print(f"    Full Model Checkpoint:  {out_dir / 'best_pretrain_model.pt'}")
    print(f"    Pretrained Backbone:    {out_dir / 'pretrained_backbone.pt'}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_experiment()

