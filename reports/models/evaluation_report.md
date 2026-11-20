# ClaimRisk AI - Model Evaluation & Threshold Analysis Report

## 1. Executive Summary

- **Evaluation Dataset**: 4,000 test claims (1,154 flagged for review, 28.8% prevalence).
- **Selected Champion Model**: **`HistGradientBoosting`** with **PR-AUC = 0.7526** and **ROC-AUC = 0.8486**.
- **Recommended Operating Threshold**: `0.475` (F1-Optimal) or `0.275` (Cost-Optimal).

---

## 2. Multi-Model Performance Benchmark Table

| Model | PR-AUC (Priority) | ROC-AUC | F1-Score (0.50) | Recall | Precision | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HistGradientBoosting 🏆** | **0.7526** | 0.8486 | 0.6502 | 0.7279 | 0.5874 | 0.1510 |
| **LogisticRegression** | **0.7517** | 0.8502 | 0.6504 | 0.7054 | 0.6034 | 0.1506 |
| **PyTorch_NN** | **0.7506** | 0.8500 | 0.6548 | 0.7201 | 0.6004 | 0.1538 |
| **RandomForest** | **0.7391** | 0.8404 | 0.6370 | 0.6447 | 0.6294 | 0.1448 |

---

## 3. Threshold Analysis & Review Prioritization

Operating the model at different probability cutoffs allows audit teams to dynamically adjust review workloads:

| Operating Policy | Decision Threshold | Recall (Coverage) | Precision | Review Workload | Simulated Cost |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Default (Balanced)** | `0.5` | 72.79% | 58.74% | 35.75% | $92,270.00 |
| **F1-Optimal** | `0.475` | 74.96% | 57.82% | 37.40% | $89,420.00 |
| **Cost-Optimal** | `0.275` | 89.51% | 44.47% | 58.07% | $79,855.00 |

---

## 4. Deliverable Visualizations
- `model_comparison_roc.png`: Multi-model ROC comparison with area-under-curve ratings.
- `model_comparison_pr.png`: Precision-Recall curve highlighting audit efficiency over baseline.
- `threshold_analysis.png`: Trade-off curves across precision, recall, auditor workload, and cost.
- `confusion_matrices.png`: Clean confusion matrices for each model.
- `calibration_curves.png`: Reliability diagram measuring probability calibration accuracy.
