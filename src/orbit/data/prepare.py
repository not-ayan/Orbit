"""
Data preparation script for ORBIT.
Loads, validates, scales, and prepares sliding window datasets for SWaT and WADI.
"""

import argparse
from pathlib import Path
import sys

# Ensure src is in python path
src_dir = Path(__file__).resolve().parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from orbit.data.pipeline import TelemetryPipeline


def main():
    parser = argparse.ArgumentParser(description="ORBIT Data Pipeline Preparation")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["wadi", "swat", "all"],
        help="Dataset to prepare (default: all)",
    )
    parser.add_argument("--window-size", type=int, default=128, help="Sequence length L")
    parser.add_argument("--stride-train", type=int, default=32, help="Stride S for train/val")
    parser.add_argument("--stride-test", type=int, default=32, help="Stride S for test")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    parser.add_argument("--scale-method", type=str, default="standard", choices=["standard", "minmax"])
    parser.add_argument("--max-train-rows", type=int, default=None, help="Optional max train rows for fast testing")
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional max test rows for fast testing")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoaders")
    args = parser.parse_args()

    datasets = ["wadi", "swat"] if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        print(f"\n{'='*25} RUNNING PIPELINE: {ds.upper()} {'='*25}")
        if ds == "wadi":
            data_dir = Path("datasets/WaDi.A2_19 Nov 2019/WADI.A2_19 Nov 2019")
        else:
            data_dir = Path("datasets/SWaT.A8_June 2021/For Mark")

        pipeline = TelemetryPipeline(
            dataset_name=ds,
            data_dir=data_dir,
            window_size=args.window_size,
            stride_train=args.stride_train,
            stride_test=args.stride_test,
            val_ratio=args.val_ratio,
            scale_method=args.scale_method,
            batch_size=args.batch_size,
        )

        train_ds, val_ds, test_ds = pipeline.run(
            normal_max_rows=args.max_train_rows,
            attack_max_rows=args.max_test_rows,
        )

        train_loader, val_loader, test_loader = pipeline.get_dataloaders(train_ds, val_ds, test_ds)

        # Verify a sample batch
        sample_x, sample_y = next(iter(train_loader))
        print(f"\n[Verification] Sample Train Batch: X shape = {sample_x.shape}, y shape = {sample_y.shape}")
        sample_tx, sample_ty = next(iter(test_loader))
        print(f"[Verification] Sample Test Batch:  X shape = {sample_tx.shape}, y shape = {sample_ty.shape}")
        print(f"[Verification] Sample Test Batch Anomaly Count: {sample_ty.sum().item()} / {len(sample_ty)}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

