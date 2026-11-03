"""
Anomaly detection scoring and evaluation pipeline.

Supports:
1. Reconstruction of continuous 1D timestamp scores from overlapping sliding windows.
2. Honest point-wise metrics (AUROC, PR-AUC, F1, Precision, Recall) on raw timestamps.
3. Event-level anomaly detection metrics (contiguous interval matching).
4. Zero test-label leakage: threshold is selected strictly from normal validation/calibration data.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from orbit.data.windowing import SlidingWindowDataset
from orbit.evaluation.metrics import (
    AnomalyMetrics,
    select_validation_threshold,
    reconstruct_timestamp_scores,
    evaluate_timestamp_predictions,
)


class AnomalyEvaluator:
    """
    Evaluates reconstruction-based anomaly detection models with timestamp-level resolution.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        aggregation: str = "mean",  # 'mean' or 'max' across overlapping windows
        matching_rule: str = "overlap",  # 'overlap' or 'iou'
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()
        self.aggregation = aggregation
        self.matching_rule = matching_rule

    @torch.no_grad()
    def compute_step_errors(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes per-timestep reconstruction errors for all windows in dataloader.

        Returns:
            step_errors: Array of shape (num_windows, window_size) with per-step MSE.
            window_scores: Array of shape (num_windows,) aggregated across window.
            window_labels: Array of shape (num_windows,) binary labels.
        """
        # Strictly enforce inference validation before scoring
        if hasattr(self.model, "validate_for_inference"):
            self.model.validate_for_inference()

        all_step_errors = []
        all_window_scores = []
        all_labels = []

        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            # Reconstruct batch: (B, L, F)
            recon_x = self.model(batch_x)
            # Per-timestep squared error across features: (B, L)
            per_step_err = ((batch_x - recon_x) ** 2).mean(dim=-1)

            if self.aggregation == "mean":
                win_scores = per_step_err.mean(dim=-1)
            elif self.aggregation == "max":
                win_scores = per_step_err.max(dim=-1).values
            else:
                win_scores = per_step_err.mean(dim=-1)

            all_step_errors.append(per_step_err.cpu().numpy())
            all_window_scores.append(win_scores.cpu().numpy())

            if batch_y.ndim > 1:
                batch_y_win = (batch_y.sum(dim=-1) > 0).long()
            else:
                batch_y_win = batch_y

            all_labels.append(batch_y_win.numpy())

        step_errors = np.concatenate(all_step_errors, axis=0) if all_step_errors else np.empty((0, 0))
        window_scores = np.concatenate(all_window_scores, axis=0) if all_window_scores else np.empty(0)
        window_labels = np.concatenate(all_labels, axis=0) if all_labels else np.empty(0)

        return step_errors, window_scores, window_labels

    def compute_anomaly_scores(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Backward-compatible helper returning window-level scores and labels."""
        _, window_scores, window_labels = self.compute_step_errors(dataloader)
        return window_scores, window_labels

    def evaluate(
        self,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        external_threshold: Optional[float] = None,
        threshold_method: str = "percentile",
        percentile: float = 99.0,
        k_sigma: float = 3.0,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> AnomalyMetrics:
        """
        Selects threshold on normal validation/calibration data and evaluates test split.
        Performs continuous timestamp-level reconstruction and evaluation with matched semantics.
        Supports external thresholding (for strict zero-shot without target data).
        """
        if test_loader is None:
            raise ValueError("test_loader cannot be None in AnomalyEvaluator.evaluate()")

        print(f"Evaluating anomaly detection performance (aggregation='{self.aggregation}')...")

        # 1. Determine threshold (from calibration data or external source)
        if external_threshold is not None:
            threshold = float(external_threshold)
            threshold_label = "external_source_threshold"
            val_window_scores = np.empty(0)
            print(f"Using external threshold: {threshold:.6f}")
        elif val_loader is not None:
            val_step_errors, val_window_scores, _ = self.compute_step_errors(val_loader)
            val_dataset = val_loader.dataset if val_loader else None
            val_raw_ds = val_dataset.dataset if isinstance(val_dataset, Subset) else val_dataset

            if isinstance(val_raw_ds, SlidingWindowDataset) and val_step_errors.ndim == 2:
                val_timestamp_scores, val_mask = reconstruct_timestamp_scores(
                    window_step_scores=val_step_errors,
                    window_size=val_raw_ds.window_size,
                    stride=val_raw_ds.stride,
                    total_timesteps=len(val_raw_ds.data),
                    aggregation=self.aggregation,
                )
                cal_scores = val_timestamp_scores[val_mask] if np.any(val_mask) else val_step_errors.ravel()
            elif val_step_errors.ndim == 2:
                cal_scores = val_step_errors.ravel()
            else:
                cal_scores = val_window_scores

            threshold = select_validation_threshold(
                cal_scores, method=threshold_method, percentile=percentile, k_sigma=k_sigma
            )
            threshold_label = f"cal_{threshold_method}_p{percentile}"
            print(f"Validation threshold selected ({threshold_method}, p={percentile}%): {threshold:.6f}")
        else:
            raise ValueError(
                "Neither val_loader nor external_threshold was provided to evaluate(). "
                "In strict zero-shot mode without target data, supply external_threshold."
            )

        # 2. Test split: extract step errors and ground truth
        test_step_errors, test_window_scores, test_window_labels = self.compute_step_errors(test_loader)

        # 3. Resolve underlying test dataset dimensions for timestamp reconstruction
        dataset = test_loader.dataset
        raw_ds = dataset.dataset if isinstance(dataset, Subset) else dataset

        if isinstance(raw_ds, SlidingWindowDataset):
            total_timesteps = len(raw_ds.data)
            window_size = raw_ds.window_size
            stride = raw_ds.stride
            raw_labels = raw_ds.labels.numpy() if raw_ds.labels is not None else None
        else:
            total_timesteps = len(test_window_scores)
            window_size = test_step_errors.shape[1] if test_step_errors.ndim == 2 else 1
            stride = 1
            raw_labels = None

        if raw_labels is not None and test_step_errors.ndim == 2:
            # Reconstruct continuous 1D timestamp scores
            timestamp_scores, coverage_mask = reconstruct_timestamp_scores(
                window_step_scores=test_step_errors,
                window_size=window_size,
                stride=stride,
                total_timesteps=total_timesteps,
                aggregation=self.aggregation,
            )

            metrics = evaluate_timestamp_predictions(
                timestamp_scores=timestamp_scores,
                timestamp_labels=raw_labels,
                threshold=threshold,
                threshold_method=threshold_label,
                matching_rule=self.matching_rule,
                coverage_mask=coverage_mask,
            )
        else:
            # Fallback to window scores if raw labels are not aligned
            metrics = evaluate_timestamp_predictions(
                timestamp_scores=test_window_scores,
                timestamp_labels=test_window_labels,
                threshold=threshold,
                threshold_method=threshold_label,
                matching_rule=self.matching_rule,
            )

        print(f"Evaluation Results (Timestamp-Level):")
        print(f"  AUROC:    {metrics.timestamp_auroc:.4f}")
        print(f"  PR-AUC:   {metrics.timestamp_pr_auc:.4f}")
        print(f"  F1-Score: {metrics.timestamp_f1:.4f} (Precision: {metrics.timestamp_precision:.4f}, Recall: {metrics.timestamp_recall:.4f})")
        print(f"  Events:   Detected {metrics.num_detected_events} / {metrics.num_gt_events} events (Event F1: {metrics.event_f1:.4f})")
        print(f"  Coverage: {metrics.covered_timesteps} / {metrics.total_timesteps} timesteps ({metrics.coverage_fraction*100:.1f}%)")
        print(f"  Matrix:   TP={metrics.tp}, FP={metrics.fp}, FN={metrics.fn}, TN={metrics.tn}")

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            with open(out_path / "metrics.json", "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)

            dist_stats = {
                "val_score_mean": float(np.mean(val_window_scores)),
                "val_score_std": float(np.std(val_window_scores)),
                "val_threshold": float(threshold),
                "covered_timesteps": metrics.covered_timesteps,
                "total_timesteps": metrics.total_timesteps,
                "coverage_fraction": metrics.coverage_fraction,
            }
            with open(out_path / "score_distribution.json", "w") as f:
                json.dump(dist_stats, f, indent=2)

        return metrics


__all__ = [
    "AnomalyEvaluator",
]
