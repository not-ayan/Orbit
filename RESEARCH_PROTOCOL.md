# ORBIT Research Protocol

**Project**: ORBIT (Operational Recognition of Behavioural Intelligence with Transformers)  
**Standard**: Formal Scientific Protocol for Few-Shot Industrial Telemetry Transfer  
**Date**: 2026-11-03  
**Version**: 1.0.0

---

## 1. Central Research Question & Scope

### 1.1 Research Question
> *“Can self-supervised temporal pretraining improve anomaly detection when a multivariate time-series model adapts to a previously unseen industrial system using only a small amount of target-domain data?”*

### 1.2 Domain Scope Discipline
The initial benchmark datasets evaluated in ORBIT are:
- **WADI (Water Distribution Testbed)**: 92 physical sensors/actuators.
- **SWaT (Secure Water Treatment Testbed)**: 52 physical sensors/actuators.

> **Crucial Scientific Limitation**:  
> Both WADI and SWaT represent complex, cyber-physical water treatment and distribution infrastructures. While transferring between them validates cross-plant adaptation between heterogeneous systems with differing dimensions ($92 \to 52$ features), **success on SWaT and WADI must NOT be claimed as universal or cross-industry generalization** (e.g. to aerospace, power grids, or manufacturing).

---

## 2. Target Data Budget Accounting

To prevent misleading claims in few-shot industrial anomaly detection, ORBIT enforces strict and transparent data budget accounting.

### 2.1 Formal Budget Definitions
Let $N_{\text{normal}}$ denote the total pool of eligible normal operating windows available in the target domain training split. For a declared target budget $B \in [0.0, 100.0]\%$, the total budgeted window count is:
$$N_{\text{budget}} = \lceil N_{\text{normal}} \times (B / 100) \rceil$$

Under ORBIT's `STRICT_INCLUSIVE` protocol, **$N_{\text{budget}}$ is an absolute ceiling on ALL target normal data accessed**. Target data cannot be secretly consumed for hyperparameter tuning, early stopping, or threshold calibration without being charged to the budget.

### 2.2 Chronological Partitioning Rules
Target windows are selected strictly in chronological order from the beginning of the normal operating period:
1. **Target Adaptation ($N_{\text{train}}$)**: Allocated $70\%$ of budgeted windows ($\ge 1$). Used for gradient updates.
2. **Target Validation ($N_{\text{val}}$)**: Allocated $15\%$ of budgeted windows. Used for early stopping and model checkpoint selection.
   - *Special Case ($N_{\text{budget}} \le 3$)*: When $N_{\text{budget}} < 4$ (e.g. at $1\%$ budget where only 1–3 windows are available), no separate validation split is carved out. Training runs for fixed epochs to prevent training data starvation.
3. **Target Calibration ($N_{\text{cal}}$)**: Allocated remaining budgeted windows ($\ge 1$). Used strictly for unsupervised anomaly threshold selection.

### 2.3 Raw-Timeline Disjoint Partitioning & Embargo Gaps
To prevent subtle information leakage where overlapping sliding windows share raw physical measurements across split boundaries:
- The target raw timeline is partitioned chronologically into disjoint subsets prior to windowing.
- An **embargo gap** of length $\ge L$ (window sequence length, default 128) is enforced between `train`, `val`, and `cal` raw partitions.
- No window generated in adaptation can contain any raw observations belonging to validation or calibration.
- **Scaler Fitting Policy**: Normalization statistics (means, standard deviations, medians) and constant feature masks are fitted **strictly on the adaptation split**. Unseen validation, calibration, and test intervals are transformed deterministically without refitting.

| Declared Budget | Total Target Windows Accessed | Adaptation ($N_{\text{train}}$) | Validation ($N_{\text{val}}$) | Calibration ($N_{\text{cal}}$) |
| :--- | :--- | :--- | :--- | :--- |
| **0% (Zero-Shot)** | 0 | 0 | 0 | 0 (Requires shared schema or mapping) |
| **1% (Few-Shot)** | 22 | 15 | 3 | 4 |
| **5% (Few-Shot)** | 106 | 74 | 15 | 17 |
| **10% (Few-Shot)** | 212 | 148 | 31 | 33 |
| **100% (Full)** | 2,112 | 1,478 | 316 | 318 |

---

## 3. Strict Test-Label Isolation & Zero Leakage

1. **No Target Test Labels in Pre-Evaluation**:
   Target test telemetry ($X_{\text{test}}$) and test labels ($Y_{\text{test}}$) are strictly quarantined until all training, validation, early stopping, and threshold calibration are finalized.
2. **Threshold Selection on Calibration Data Only**:
   The anomaly detection threshold $\theta$ is selected strictly from the reconstruction error distribution of normal calibration data ($X_{\text{cal}}$):
   - **Percentile Threshold**: $\theta = \text{Percentile}(S_{\text{cal}}, 99.0)$
   - **Extreme Value / Gaussian Threshold**: $\theta = \mu_{\text{cal}} + 3\sigma_{\text{cal}}$
   Test labels are NEVER used to optimize or tune the anomaly threshold.

---

## 4. Continuous Timestamp-Level Reconstruction

Sliding window evaluations often introduce evaluation bias by reducing multi-step windows to a single binary label (`ANY` positive policy) and reporting window metrics as point-wise performance. ORBIT resolves this by projecting reconstruction errors back to continuous physical time.

### 4.1 Timestamp Mapping
Given an autoencoder producing reconstructions $\hat{X}_i \in \mathbb{R}^{L \times F}$ for window $i \in [0, W-1]$, the per-timestep squared error across $F$ features is:
$$e_{i, t} = \frac{1}{F} \sum_{f=1}^F (x_{i, t, f} - \hat{x}_{i, t, f})^2, \quad t \in [0, L-1]$$

Each window $i$ begins at sample index $i \cdot S$, where $S$ is stride. For each continuous global timestamp $k \in [0, T-1]$, the aggregated timestamp anomaly score $S_k$ is computed by arithmetic mean over all covering windows:
$$S_k = \frac{\sum_{i \text{ covering } k} e_{i, k - i \cdot S}}{\sum_{i \text{ covering } k} 1}$$

### 4.2 Edge Coverage Handling
Timestamps at the trailing edge not covered by any full window ($k \ge (W-1)S + L$) are explicitly flagged in `coverage_mask`. Evaluation metrics are computed strictly over covered timestamps ($T_{\text{covered}}$), and the coverage fraction ($T_{\text{covered}} / T$) is recorded in the experiment metadata.

---

## 5. Event-Level Anomaly Detection Metrics

In operational industrial SCADA systems, operators respond to alarm incidents rather than isolated 1-second timestamps.

### 5.1 Event Extraction
Contiguous runs of positive ground-truth timestamps ($y_k = 1$) define true anomaly events $E_{\text{gt}} = [t_{\text{start}}, t_{\text{end}}]$. Contiguous runs of alarms ($S_k \ge \theta$) define predicted alarm episodes $E_{\text{pred}} = [t_{\text{start}}, t_{\text{end}}]$.

### 5.2 Strict One-to-One Bipartite Matching
To prevent metric inflation where multiple fragmented predicted alarms match the same single ground-truth event (or one broad alarm credits multiple distinct events), ORBIT enforces **deterministic one-to-one maximum overlap matching**:
1. Each ground-truth event $G_i$ can match at most one predicted alarm $P_j$.
2. Each predicted alarm episode $P_j$ can match at most one ground-truth event $G_i$.
3. Matching pairs are assigned greedily in descending order of intersection overlap (or IoU).
4. Unmatched ground-truth events are recorded as False Negatives ($FN_{\text{event}}$); unmatched predicted alarms are recorded as False Positives ($FP_{\text{event}}$).

$$\text{Event Recall} = \frac{TP_{\text{event}}}{|E_{\text{gt}}|}$$
$$\text{Event Precision} = \frac{TP_{\text{event}}}{|E_{\text{pred}}|}$$
$$\text{Event F1} = \frac{2 \cdot \text{Event Precision} \cdot \text{Event Recall}}{\text{Event Precision} + \text{Event Recall}}$$

---

## 6. Model Architectures & Baseline Controls

All models share identical chronological splits, normalization, window parameters ($L=128, S=32$), and evaluation procedures:
1. **LSTM Autoencoder Baseline**: 2-layer encoder, 2-layer decoder with latent dimension $D_{\text{latent}}=32$.
2. **Scratch Transformer Baseline**: 3 Pre-LN layers, 4 attention heads, $D_{\text{model}}=64$, feedforward dimension 256. Trained from scratch on the budgeted target data.
3. **ORBIT Pretrained + Adapter**: Pretrained temporal backbone (trained on WADI with 20% block masking). Backbone is frozen; residual bottleneck adapters ($D_{\text{adapter}}=16$, $8.35\%$ trainable parameters) and projection layers adapt to target domain.
4. **ORBIT Full Fine-Tuning**: Pretrained backbone with all parameters updated end-to-end on budgeted target data.

---

## 7. Multi-Seed Replication Protocol

All transfer and few-shot benchmark experiments must be executed across at least 3 independent random seeds:
- Seed set: `[42, 43, 44]`
- Results must report individual seed outcomes and aggregate `mean ± standard deviation`.
- No single-seed cherry-picking is permitted in official benchmark tables.

