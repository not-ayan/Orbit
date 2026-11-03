"""
Evaluation metrics for industrial anomaly detection.

Provides:
1. Timestamp-level continuous scoring (reconstructed from overlapping sliding windows).
2. Point-wise metrics (AUROC, PR-AUC, F1, Precision, Recall, Confusion Matrix) on raw timestamps.
3. Event-level metrics with explicit overlap or IoU matching.
4. Optional window-level metrics (clearly labeled with window_ prefix).
5. Strict zero-test-label leakage threshold selection on calibration/validation splits.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


@dataclass
class AnomalyMetrics:
    # Continuous Timestamp-level threshold-free metrics
    timestamp_auroc: float
    timestamp_pr_auc: float

    # Continuous Timestamp-level thresholded metrics
    threshold: float
    threshold_method: str
    timestamp_precision: float
    timestamp_recall: float
    timestamp_f1: float
    tp: int
    fp: int
    tn: int
    fn: int

    # Event-level detection metrics
    num_gt_events: int
    num_predicted_events: int
    num_detected_events: int
    event_precision: float
    event_recall: float
    event_f1: float
    event_matching_rule: str

    # Coverage and dataset accounting
    total_timesteps: int
    covered_timesteps: int
    coverage_fraction: float

    # Optional window-level metrics (for baseline comparison)
    window_auroc: Optional[float] = None
    window_pr_auc: Optional[float] = None
    window_f1: Optional[float] = None

    # Backward compatibility properties
    @property
    def auroc(self) -> float:
        return self.timestamp_auroc

    @property
    def pr_auc(self) -> float:
        return self.timestamp_pr_auc

    @property
    def precision(self) -> float:
        return self.timestamp_precision

    @property
    def recall(self) -> float:
        return self.timestamp_recall

    @property
    def f1(self) -> float:
        return self.timestamp_f1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_validation_threshold(
    val_scores: np.ndarray,
    method: str = "percentile",
    percentile: float = 99.0,
    k_sigma: float = 3.0,
) -> float:
    """
    Selects anomaly threshold strictly on validation/calibration normal reconstruction scores.
    Zero test-label leakage guaranteed.

    Args:
        val_scores: 1D array of anomaly scores on validation normal data.
        method: 'percentile', 'mean_std', or 'max'.
        percentile: Percentile value (e.g. 99.0 = 99th percentile).
        k_sigma: Multiplier for mean + k * std.
    """
    if len(val_scores) == 0:
        raise ValueError("Cannot select threshold from empty validation scores.")

    if method == "percentile":
        threshold = float(np.percentile(val_scores, percentile))
    elif method == "mean_std":
        mean = float(np.mean(val_scores))
        std = float(np.std(val_scores))
        threshold = mean + k_sigma * std
    elif method == "max":
        threshold = float(np.max(val_scores))
    else:
        raise ValueError(f"Unknown threshold method: {method}")

    return threshold


def reconstruct_timestamp_scores(
    window_step_scores: np.ndarray,
    window_size: int,
    stride: int,
    total_timesteps: int,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maps per-timestep errors from overlapping sliding windows back to a continuous 1D timeline.

    Args:
        window_step_scores: Array of shape (num_windows, window_size) containing per-step errors.
        window_size: Window sequence length L.
        stride: Window stride S.
        total_timesteps: Total length T of the raw continuous telemetry.
        aggregation: 'mean' (arithmetic average across overlapping windows) or 'max'.

    Returns:
        timestamp_scores: 1D array of shape (total_timesteps,).
        coverage_mask: 1D boolean array indicating timestamps covered by >= 1 window.
    """
    num_windows, L = window_step_scores.shape
    if L != window_size:
        raise ValueError(f"window_step_scores sequence length {L} != expected {window_size}")

    coverage_counts = np.zeros(total_timesteps, dtype=np.int32)

    if aggregation == "mean":
        score_accumulator = np.zeros(total_timesteps, dtype=np.float64)
        for i in range(num_windows):
            start = i * stride
            end = min(start + L, total_timesteps)
            length = end - start
            score_accumulator[start:end] += window_step_scores[i, :length]
            coverage_counts[start:end] += 1

        coverage_mask = coverage_counts > 0
        timestamp_scores = np.zeros(total_timesteps, dtype=np.float32)
        timestamp_scores[coverage_mask] = (
            score_accumulator[coverage_mask] / coverage_counts[coverage_mask]
        ).astype(np.float32)

    elif aggregation == "max":
        timestamp_scores = np.full(total_timesteps, -np.inf, dtype=np.float32)
        for i in range(num_windows):
            start = i * stride
            end = min(start + L, total_timesteps)
            length = end - start
            timestamp_scores[start:end] = np.maximum(
                timestamp_scores[start:end], window_step_scores[i, :length]
            )
            coverage_counts[start:end] += 1

        coverage_mask = coverage_counts > 0
        timestamp_scores[~coverage_mask] = 0.0

    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Supported: 'mean', 'max'.")

    return timestamp_scores, coverage_mask


def find_contiguous_events(binary_sequence: np.ndarray) -> List[Tuple[int, int]]:
    """
    Finds contiguous segments of 1s in a binary array.
    Returns list of (start_idx, end_idx) tuples inclusive.
    """
    events = []
    in_event = False
    start = 0

    for i, val in enumerate(binary_sequence):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append((start, i - 1))

    if in_event:
        events.append((start, len(binary_sequence) - 1))

    return events


def compute_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    matching_rule: str = "overlap",
    iou_threshold: float = 0.1,
    min_event_duration: int = 1,
) -> Tuple[int, int, float, float, float]:
    """
    Calculates event-level detection performance with strict ONE-TO-ONE matching.
    Each ground truth event can match at most one predicted alarm episode, and vice versa.
    Candidate overlapping pairs are matched deterministically by maximum overlap/IoU.

    Args:
        y_true: 1D ground truth binary array on timestamps.
        y_pred: 1D predicted binary alarm array on timestamps.
        matching_rule: 'overlap' (any overlap detects event) or 'iou' (intersection over union >= threshold).
        iou_threshold: Minimum IoU required if matching_rule == 'iou'.
        min_event_duration: Minimum consecutive timestamps to qualify as an event.

    Returns:
        num_gt_events: Total ground truth anomaly episodes.
        num_detected_events: Ground truth events successfully triggered (TP).
        event_precision: Fraction of predicted alarm episodes that hit true events.
        event_recall: Fraction of ground truth events detected.
        event_f1: Harmonic mean of event precision and recall.
    """
    gt_events = find_contiguous_events(y_true)
    pred_events = find_contiguous_events(y_pred)

    if min_event_duration > 1:
        gt_events = [e for e in gt_events if (e[1] - e[0] + 1) >= min_event_duration]
        pred_events = [e for e in pred_events if (e[1] - e[0] + 1) >= min_event_duration]

    num_gt = len(gt_events)
    num_pred = len(pred_events)

    if num_gt == 0 and num_pred == 0:
        return 0, 0, 1.0, 1.0, 1.0
    if num_gt == 0:
        return 0, 0, 0.0, 1.0, 0.0
    if num_pred == 0:
        return num_gt, 0, 1.0, 0.0, 0.0

    # 1. Find all candidate overlapping pairs with overlap/IoU score
    candidate_matches = []
    for g_idx, (g_start, g_end) in enumerate(gt_events):
        g_len = g_end - g_start + 1
        for p_idx, (p_start, p_end) in enumerate(pred_events):
            p_len = p_end - p_start + 1
            overlap_start = max(g_start, p_start)
            overlap_end = min(g_end, p_end)
            overlap_len = max(0, overlap_end - overlap_start + 1)

            if overlap_len > 0:
                union_len = g_len + p_len - overlap_len
                iou = overlap_len / union_len if union_len > 0 else 0.0

                if matching_rule == "overlap":
                    candidate_matches.append((overlap_len, g_idx, p_idx))
                elif matching_rule == "iou":
                    if iou >= iou_threshold:
                        candidate_matches.append((iou, g_idx, p_idx))

    # 2. Maximum-overlap deterministic one-to-one matching
    candidate_matches.sort(key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()

    for weight, g_idx, p_idx in candidate_matches:
        if g_idx not in matched_gt and p_idx not in matched_pred:
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)

    tp_events = len(matched_gt)
    event_precision = (tp_events / num_pred) if num_pred > 0 else 0.0
    event_recall = (tp_events / num_gt) if num_gt > 0 else 0.0

    if event_precision + event_recall > 0:
        event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall)
    else:
        event_f1 = 0.0

    return num_gt, tp_events, event_precision, event_recall, event_f1


def evaluate_timestamp_predictions(
    timestamp_scores: np.ndarray,
    timestamp_labels: np.ndarray,
    threshold: float,
    threshold_method: str = "percentile",
    matching_rule: str = "overlap",
    coverage_mask: Optional[np.ndarray] = None,
) -> AnomalyMetrics:
    """
    Evaluates continuous timestamp anomaly detection metrics.
    """
    if coverage_mask is not None:
        eval_scores = timestamp_scores[coverage_mask]
        eval_labels = timestamp_labels[coverage_mask]
        covered_count = int(coverage_mask.sum())
    else:
        eval_scores = timestamp_scores
        eval_labels = timestamp_labels
        covered_count = len(timestamp_scores)

    total_count = len(timestamp_scores)
    coverage_frac = covered_count / max(1, total_count)

    # Threshold-free metrics
    try:
        auroc = float(roc_auc_score(eval_labels, eval_scores))
    except ValueError:
        auroc = 0.5

    try:
        pr_auc = float(average_precision_score(eval_labels, eval_scores))
    except ValueError:
        pr_auc = 0.0

    # Thresholded point-wise metrics
    y_pred = (eval_scores >= threshold).astype(int)
    prec = float(precision_score(eval_labels, y_pred, zero_division=0))
    rec = float(recall_score(eval_labels, y_pred, zero_division=0))
    f1 = float(f1_score(eval_labels, y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(eval_labels, y_pred, labels=[0, 1]).ravel()

    # Event metrics
    pred_events = find_contiguous_events(y_pred)
    num_pred = len(pred_events)
    num_gt, num_detected, ev_prec, ev_rec, ev_f1 = compute_event_metrics(
        y_true=eval_labels,
        y_pred=y_pred,
        matching_rule=matching_rule,
    )

    return AnomalyMetrics(
        timestamp_auroc=auroc,
        timestamp_pr_auc=pr_auc,
        threshold=float(threshold),
        threshold_method=threshold_method,
        timestamp_precision=prec,
        timestamp_recall=rec,
        timestamp_f1=f1,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        num_gt_events=num_gt,
        num_predicted_events=num_pred,
        num_detected_events=num_detected,
        event_precision=ev_prec,
        event_recall=ev_rec,
        event_f1=ev_f1,
        event_matching_rule=matching_rule,
        total_timesteps=total_count,
        covered_timesteps=covered_count,
        coverage_fraction=round(coverage_frac, 4),
    )


# Backward-compatible function name
def evaluate_anomaly_predictions(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    threshold_method: str = "percentile",
) -> AnomalyMetrics:
    """Wrapper supporting legacy 1D score evaluation."""
    return evaluate_timestamp_predictions(
        timestamp_scores=scores,
        timestamp_labels=labels,
        threshold=threshold,
        threshold_method=threshold_method,
    )


__all__ = [
    "AnomalyMetrics",
    "select_validation_threshold",
    "reconstruct_timestamp_scores",
    "find_contiguous_events",
    "compute_event_metrics",
    "evaluate_timestamp_predictions",
    "evaluate_anomaly_predictions",
]
