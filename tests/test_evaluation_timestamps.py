"""
Unit tests for timestamp reconstruction, edge coverage, point-wise metrics, and event matching.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from orbit.evaluation.metrics import (
    reconstruct_timestamp_scores,
    find_contiguous_events,
    compute_event_metrics,
    evaluate_timestamp_predictions,
)
from orbit.evaluation.evaluator import AnomalyEvaluator
from orbit.data.windowing import SlidingWindowDataset


def test_reconstruct_timestamp_scores_mean_overlap():
    # 4 windows of length 4, stride 2 -> covers indices:
    # w0: 0..3 (scores [1, 1, 1, 1])
    # w1: 2..5 (scores [2, 2, 2, 2])
    # Total timesteps: 8
    # Coverage:
    # 0, 1: covered by w0 only -> mean 1.0
    # 2, 3: covered by w0 and w1 -> mean (1 + 2)/2 = 1.5
    # 4, 5: covered by w1 only -> mean 2.0
    # 6, 7: uncovered -> count 0, score 0.0
    window_scores = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ], dtype=np.float32)

    total_T = 8
    scores, mask = reconstruct_timestamp_scores(
        window_step_scores=window_scores,
        window_size=4,
        stride=2,
        total_timesteps=total_T,
        aggregation="mean",
    )

    assert len(scores) == total_T
    assert np.all(mask[:6])
    assert not np.any(mask[6:])

    expected = np.array([1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(scores, expected, atol=1e-5)


def test_reconstruct_timestamp_scores_max_overlap():
    window_scores = np.array([
        [1.0, 3.0, 1.0, 1.0],
        [2.0, 4.0, 2.0, 2.0],
    ], dtype=np.float32)

    total_T = 6
    scores, mask = reconstruct_timestamp_scores(
        window_step_scores=window_scores,
        window_size=4,
        stride=2,
        total_timesteps=total_T,
        aggregation="max",
    )

    # index 2: max(w0[2]=1.0, w1[0]=2.0) = 2.0
    # index 3: max(w0[3]=1.0, w1[1]=4.0) = 4.0
    assert scores[2] == 2.0
    assert scores[3] == 4.0


def test_short_anomaly_detection_resolution():
    """
    Verifies that an anomaly significantly shorter than the window size
    is localized at the timestamp level rather than inflating the entire window.
    """
    total_T = 100
    L = 32
    S = 8
    # Ground truth: anomaly only at timestamps 40..44 (length 5 << window 32)
    y_true = np.zeros(total_T, dtype=np.int64)
    y_true[40:45] = 1

    # Simulate reconstruction error spike strictly at timestamps 40..44
    num_windows = (total_T - L) // S + 1
    step_scores = np.zeros((num_windows, L), dtype=np.float32)

    for i in range(num_windows):
        start = i * S
        for t in range(L):
            global_t = start + t
            if 40 <= global_t < 45:
                step_scores[i, t] = 5.0
            else:
                step_scores[i, t] = 0.1

    timestamp_scores, mask = reconstruct_timestamp_scores(
        window_step_scores=step_scores,
        window_size=L,
        stride=S,
        total_timesteps=total_T,
        aggregation="mean",
    )

    # Threshold selected at 1.0
    metrics = evaluate_timestamp_predictions(
        timestamp_scores=timestamp_scores,
        timestamp_labels=y_true,
        threshold=1.0,
        coverage_mask=mask,
    )

    # Peak scores must be at 40..44
    assert np.all(timestamp_scores[40:45] > 4.0)
    assert np.all(timestamp_scores[:40] < 0.5)
    assert metrics.timestamp_f1 > 0.95
    assert metrics.num_gt_events == 1
    assert metrics.num_detected_events == 1


def test_event_matching_overlap_and_iou():
    # 2 true events: [10..20] and [50..60]
    y_true = np.zeros(100, dtype=np.int64)
    y_true[10:21] = 1
    y_true[50:61] = 1

    # Prediction 1: [15..25] overlaps [10..20]
    # Prediction 2: [80..90] false alarm
    y_pred = np.zeros(100, dtype=np.int64)
    y_pred[15:26] = 1
    y_pred[80:91] = 1

    # Test overlap rule
    gt_count, detected, prec, rec, f1 = compute_event_metrics(
        y_true, y_pred, matching_rule="overlap"
    )
    assert gt_count == 2
    assert detected == 1  # 1 true event detected
    assert rec == 0.5     # 1 / 2
    assert prec == 0.5    # 1 valid alarm / 2 predicted alarms


def test_one_to_one_event_matching_no_double_counting():
    """
    Verifies that:
    1. One true event and two predicted alarms cannot both count as TP.
    2. Two true events and one predicted alarm can match at most one event.
    """
    # Case 1: 1 true event [10..50], 2 predicted alarm intervals [12..20] and [30..40]
    y_true_1 = np.zeros(100, dtype=np.int64)
    y_true_1[10:51] = 1

    y_pred_1 = np.zeros(100, dtype=np.int64)
    y_pred_1[12:21] = 1  # length 9
    y_pred_1[30:41] = 1  # length 11

    num_gt, detected, prec, rec, f1 = compute_event_metrics(y_true_1, y_pred_1)
    assert num_gt == 1
    assert detected == 1  # exactly 1 true event detected
    # With 2 predicted alarms matching the SAME 1 GT event, one-to-one matching allows only 1 TP!
    # The other predicted alarm is an FP, so precision must be 1 / 2 = 0.5!
    assert prec == 0.5
    assert rec == 1.0

    # Case 2: 2 true events [10..20] and [30..40], 1 predicted alarm [5..45] covering both
    y_true_2 = np.zeros(100, dtype=np.int64)
    y_true_2[10:21] = 1
    y_true_2[30:41] = 1

    y_pred_2 = np.zeros(100, dtype=np.int64)
    y_pred_2[5:46] = 1

    num_gt2, detected2, prec2, rec2, f12 = compute_event_metrics(y_true_2, y_pred_2)
    assert num_gt2 == 2
    # The 1 predicted alarm can match at most ONE of the true events
    assert detected2 == 1
    assert rec2 == 0.5
    assert prec2 == 1.0


def test_threshold_timestamp_score_scale_alignment():
    """
    Verifies that AnomalyEvaluator selects threshold on timestamp-level scores
    consistent with test score semantics, and supports external thresholds.
    """
    from orbit.evaluation.evaluator import AnomalyEvaluator

    class DummyModel(nn.Module):
        def forward(self, x):
            # Returns input unchanged -> 0 error
            return x

    model = DummyModel()
    evaluator = AnomalyEvaluator(model, aggregation="mean")

    # Create synthetic test dataset
    test_data = np.random.randn(200, 4).astype(np.float32)
    test_labels = np.zeros(200, dtype=np.int64)
    test_ds = SlidingWindowDataset(test_data, labels=test_labels, window_size=32, stride=16)
    test_loader = DataLoader(test_ds, batch_size=4)

    # Test with external threshold
    metrics = evaluator.evaluate(
        val_loader=None,
        test_loader=test_loader,
        external_threshold=0.5,
    )
    assert metrics.threshold == 0.5
    assert metrics.threshold_method == "external_source_threshold"
