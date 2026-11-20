"""
ClaimRisk AI - Model Evaluation & Threshold Analysis Module.

Performs rigorous evaluation across all candidate models and champion model:
- Precision, Recall, F1-Score, PR-AUC, ROC-AUC, Brier score.
- Comprehensive Decision Threshold Sweep (0.05 - 0.95) for review prioritization.
- Cost-sensitive analysis balancing manual audit expense against missed billing risk.
- Publication-quality visualizations:
    1. model_comparison_roc.png
    2. model_comparison_pr.png
    3. threshold_analysis.png
    4. confusion_matrices.png
    5. calibration_curves.png
- Comprehensive JSON and Markdown evaluation reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimrisk.evaluate")


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_audit: float = 25.0,
    cost_missed_risk: float = 180.0,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Perform decision threshold sweep evaluating operational triage trade-offs.
    
    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted positive class risk probabilities.
        cost_audit: Average cost ($) to perform manual clinical/billing audit.
        cost_missed_risk: Average financial loss/exposure ($) of an undetected at-risk claim.
        thresholds: Array of decision threshold values.
        
    Returns:
        Dictionary containing metric curves, optimal thresholds, and cost curves.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 37)

    threshold_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    review_workload_list = []
    total_cost_list = []

    n_samples = len(y_true)

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        workload = (tp + fp) / n_samples  # Percentage of claims sent to manual review

        # Operational Cost Model:
        # Audit Cost: Paid for all reviewed claims (TP + FP)
        # Missed Risk Cost: Penalty for unreviewed high-risk claims (FN)
        total_cost = (tp + fp) * cost_audit + fn * cost_missed_risk

        threshold_list.append(round(float(thresh), 4))
        precision_list.append(round(float(p), 4))
        recall_list.append(round(float(r), 4))
        f1_list.append(round(float(f1), 4))
        review_workload_list.append(round(float(workload), 4))
        total_cost_list.append(round(float(total_cost), 2))

    # Identify key optimal operating points
    best_f1_idx = int(np.argmax(f1_list))
    best_cost_idx = int(np.argmin(total_cost_list))

    # Standard default baseline 0.50
    default_idx = int(np.argmin(np.abs(np.array(threshold_list) - 0.50)))

    return {
        "thresholds": threshold_list,
        "precision": precision_list,
        "recall": recall_list,
        "f1": f1_list,
        "review_workload": review_workload_list,
        "total_cost": total_cost_list,
        "optimal_points": {
            "f1_optimal": {
                "threshold": threshold_list[best_f1_idx],
                "f1": f1_list[best_f1_idx],
                "precision": precision_list[best_f1_idx],
                "recall": recall_list[best_f1_idx],
                "workload": review_workload_list[best_f1_idx],
                "cost": total_cost_list[best_f1_idx],
            },
            "cost_optimal": {
                "threshold": threshold_list[best_cost_idx],
                "f1": f1_list[best_cost_idx],
                "precision": precision_list[best_cost_idx],
                "recall": recall_list[best_cost_idx],
                "workload": review_workload_list[best_cost_idx],
                "cost": total_cost_list[best_cost_idx],
            },
            "default_0_5": {
                "threshold": threshold_list[default_idx],
                "f1": f1_list[default_idx],
                "precision": precision_list[default_idx],
                "recall": recall_list[default_idx],
                "workload": review_workload_list[default_idx],
                "cost": total_cost_list[default_idx],
            },
        },
    }


def plot_roc_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """Generate and save multi-model ROC Curve comparison."""
    plt.figure(figsize=(9, 7))
    palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    for i, (name, (y_true, y_prob)) in enumerate(models_predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = palette[i % len(palette)]
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, linewidth=2.2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.500)")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("ROC Curves - Model Comparison Benchmark", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curve comparison to {save_path}")


def plot_precision_recall_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """Generate and save multi-model Precision-Recall Curve comparison."""
    plt.figure(figsize=(9, 7))
    palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    first_y_true = next(iter(models_predictions.values()))[0]
    baseline_prevalence = np.mean(first_y_true)

    for i, (name, (y_true, y_prob)) in enumerate(models_predictions.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        color = palette[i % len(palette)]
        plt.plot(recall, precision, label=f"{name} (PR-AUC = {pr_auc:.3f})", color=color, linewidth=2.2)

    plt.axhline(
        y=baseline_prevalence,
        color="k",
        linestyle="--",
        alpha=0.6,
        label=f"Baseline Prevalence ({baseline_prevalence:.1%})",
    )
    plt.xlabel("Recall (Coverage of At-Risk Claims)", fontsize=11, fontweight="bold")
    plt.ylabel("Precision (Auditor Review Accuracy)", fontsize=11, fontweight="bold")
    plt.title("Precision-Recall Curves - Audit Triage Benchmark", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved Precision-Recall curve comparison to {save_path}")


def plot_threshold_analysis(
    threshold_results: Dict[str, Any],
    save_path: Path,
    model_name: str = "Champion Model",
) -> None:
    """Plot multi-panel threshold trade-off analysis for audit prioritization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    threshs = threshold_results["thresholds"]
    precisions = threshold_results["precision"]
    recalls = threshold_results["recall"]
    f1s = threshold_results["f1"]
    workloads = threshold_results["review_workload"]
    costs = threshold_results["total_cost"]

    # Panel 1: Precision, Recall, F1, and Review Workload vs Threshold
    ax1.plot(threshs, precisions, label="Precision", color="#2563eb", linewidth=2.2)
    ax1.plot(threshs, recalls, label="Recall", color="#10b981", linewidth=2.2)
    ax1.plot(threshs, f1s, label="F1-Score", color="#f59e0b", linewidth=2.5)
    ax1.plot(threshs, workloads, label="Review Workload (% claims)", color="#6b7280", linestyle=":", linewidth=2)

    opt_f1 = threshold_results["optimal_points"]["f1_optimal"]
    ax1.axvline(
        opt_f1["threshold"],
        color="#f59e0b",
        linestyle="--",
        alpha=0.8,
        label=f"Max F1 ({opt_f1['threshold']:.2f})",
    )

    ax1.set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Metric Score / Proportion", fontsize=11, fontweight="bold")
    ax1.set_title(f"Classification Metrics vs Threshold ({model_name})", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=10)

    # Panel 2: Operational Cost Curve
    opt_cost = threshold_results["optimal_points"]["cost_optimal"]
    ax2.plot(threshs, costs, color="#dc2626", linewidth=2.5, label="Total Operational Cost ($)")
    ax2.axvline(
        opt_cost["threshold"],
        color="#dc2626",
        linestyle="--",
        alpha=0.8,
        label=f"Min Cost Thresh ({opt_cost['threshold']:.2f})",
    )
    ax2.scatter(
        [opt_cost["threshold"]],
        [opt_cost["cost"]],
        color="#dc2626",
        s=80,
        zorder=5,
    )

    ax2.set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Simulated Total Operational Expense ($)", fontsize=11, fontweight="bold")
    ax2.set_title("Cost-Sensitive Audit Optimization", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved threshold analysis to {save_path}")


def plot_confusion_matrices(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
    threshold: float = 0.50,
) -> None:
    """Plot confusion matrices for all models in a clean grid."""
    n_models = len(models_predictions)
    cols = min(2, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = np.ravel(axes)

    for i, (name, (y_true, y_prob)) in enumerate(models_predictions.items()):
        ax = axes[i]
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name} (Thresh={threshold})", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Clean (0)", "Audit (1)"])
        ax.set_yticklabels(["Clean (0)", "Audit (1)"])
        ax.set_xlabel("Predicted Review Label", fontweight="bold")
        ax.set_ylabel("True Audit Label", fontweight="bold")

        # Text annotations
        for r in range(2):
            for c in range(2):
                val = cm[r, c]
                color = "white" if val > cm.max() / 2 else "black"
                ax.text(c, r, f"{val:,}", ha="center", va="center", color=color, fontweight="bold", fontsize=12)

    # Hide extra subplots
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices to {save_path}")


def plot_calibration_curves(
    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Path,
) -> None:
    """Plot probability calibration curves (reliability diagrams)."""
    plt.figure(figsize=(8, 7))
    palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6"]

    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    for i, (name, (y_true, y_prob)) in enumerate(models_predictions.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        brier = brier_score_loss(y_true, y_prob)
        color = palette[i % len(palette)]
        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            color=color,
            label=f"{name} (Brier = {brier:.4f})",
        )

    plt.xlabel("Mean Predicted Risk Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Observed Fraction of Positives", fontsize=11, fontweight="bold")
    plt.title("Reliability Diagram - Probability Calibration", fontsize=13, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved calibration curves to {save_path}")


def run_comprehensive_evaluation(
    models_dir: Path,
    data_dir: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    """Execute evaluation pipeline and generate all artifacts and reports."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load test data
    X_test_df = pd.read_csv(data_dir / "X_test_processed.csv")
    y_test_df = pd.read_csv(data_dir / "y_test.csv")
    X_test = X_test_df.values.astype(np.float32)
    y_test = y_test_df.values.ravel().astype(int)

    # 2. Discover models
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No .joblib model files found in {models_dir}. Train models first.")

    models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    detailed_metrics: Dict[str, Any] = {}

    for model_path in sorted(model_files):
        if model_path.stem == "champion_model":
            continue  # Champion is an alias to one of the specific models

        name = model_path.stem
        logger.info(f"Loading and evaluating model: {name}")
        model = joblib.load(model_path)

        y_prob = model.predict_proba(X_test)[:, 1]
        models_predictions[name] = (y_test, y_prob)

        # Baseline metrics at 0.50
        y_pred = (y_prob >= 0.50).astype(int)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        detailed_metrics[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "pr_auc": round(float(average_precision_score(y_test, y_prob)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y_test, y_prob)), 4),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    # Identify champion model by PR-AUC
    champion_name = max(detailed_metrics.keys(), key=lambda k: detailed_metrics[k]["pr_auc"])
    champion_y_prob = models_predictions[champion_name][1]

    logger.info(f"Top performing model for threshold analysis: {champion_name}")

    # 3. Perform Threshold Analysis on champion model
    threshold_analysis_res = compute_threshold_metrics(
        y_true=y_test,
        y_prob=champion_y_prob,
        cost_audit=25.0,
        cost_missed_risk=180.0,
    )

    # 4. Generate Visualizations
    plot_roc_curves(models_predictions, output_dir / "model_comparison_roc.png")
    plot_precision_recall_curves(models_predictions, output_dir / "model_comparison_pr.png")
    plot_threshold_analysis(
        threshold_analysis_res,
        output_dir / "threshold_analysis.png",
        model_name=champion_name,
    )
    plot_confusion_matrices(models_predictions, output_dir / "confusion_matrices.png", threshold=0.50)
    plot_calibration_curves(models_predictions, output_dir / "calibration_curves.png")

    # 5. Compile Summary JSON
    summary = {
        "dataset_info": {
            "test_samples": len(y_test),
            "positive_count": int(np.sum(y_test == 1)),
            "negative_count": int(np.sum(y_test == 0)),
            "positive_prevalence": round(float(np.mean(y_test)), 4),
        },
        "champion_model": champion_name,
        "models_benchmarks": detailed_metrics,
        "threshold_analysis": threshold_analysis_res,
    }

    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 6. Generate Markdown Evaluation Report
    md_file = output_dir / "evaluation_report.md"
    generate_markdown_report(summary, md_file)

    logger.info(f"Evaluation complete. Reports generated at {output_dir}")
    return summary


def generate_markdown_report(summary: Dict[str, Any], save_path: Path) -> None:
    """Create comprehensive Markdown audit report."""
    ds = summary["dataset_info"]
    champ = summary["champion_model"]
    benchmarks = summary["models_benchmarks"]
    thresh_opt = summary["threshold_analysis"]["optimal_points"]

    md = f"""# ClaimRisk AI - Model Evaluation & Threshold Analysis Report

## 1. Executive Summary

- **Evaluation Dataset**: {ds['test_samples']:,} test claims ({ds['positive_count']:,} flagged for review, {ds['positive_prevalence']:.1%} prevalence).
- **Selected Champion Model**: **`{champ}`** with **PR-AUC = {benchmarks[champ]['pr_auc']:.4f}** and **ROC-AUC = {benchmarks[champ]['roc_auc']:.4f}**.
- **Recommended Operating Threshold**: `{thresh_opt['f1_optimal']['threshold']}` (F1-Optimal) or `{thresh_opt['cost_optimal']['threshold']}` (Cost-Optimal).

---

## 2. Multi-Model Performance Benchmark Table

| Model | PR-AUC (Priority) | ROC-AUC | F1-Score (0.50) | Recall | Precision | Brier Score |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for model_name, m in benchmarks.items():
        is_champ = " 🏆" if model_name == champ else ""
        md += f"| **{model_name}{is_champ}** | **{m['pr_auc']:.4f}** | {m['roc_auc']:.4f} | {m['f1_score']:.4f} | {m['recall']:.4f} | {m['precision']:.4f} | {m['brier_score']:.4f} |\n"

    md += f"""
---

## 3. Threshold Analysis & Review Prioritization

Operating the model at different probability cutoffs allows audit teams to dynamically adjust review workloads:

| Operating Policy | Decision Threshold | Recall (Coverage) | Precision | Review Workload | Simulated Cost |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Default (Balanced)** | `{thresh_opt['default_0_5']['threshold']}` | {thresh_opt['default_0_5']['recall']:.2%} | {thresh_opt['default_0_5']['precision']:.2%} | {thresh_opt['default_0_5']['workload']:.2%} | ${thresh_opt['default_0_5']['cost']:,.2f} |
| **F1-Optimal** | `{thresh_opt['f1_optimal']['threshold']}` | {thresh_opt['f1_optimal']['recall']:.2%} | {thresh_opt['f1_optimal']['precision']:.2%} | {thresh_opt['f1_optimal']['workload']:.2%} | ${thresh_opt['f1_optimal']['cost']:,.2f} |
| **Cost-Optimal** | `{thresh_opt['cost_optimal']['threshold']}` | {thresh_opt['cost_optimal']['recall']:.2%} | {thresh_opt['cost_optimal']['precision']:.2%} | {thresh_opt['cost_optimal']['workload']:.2%} | ${thresh_opt['cost_optimal']['cost']:,.2f} |

---

## 4. Deliverable Visualizations
- `model_comparison_roc.png`: Multi-model ROC comparison with area-under-curve ratings.
- `model_comparison_pr.png`: Precision-Recall curve highlighting audit efficiency over baseline.
- `threshold_analysis.png`: Trade-off curves across precision, recall, auditor workload, and cost.
- `confusion_matrices.png`: Clean confusion matrices for each model.
- `calibration_curves.png`: Reliability diagram measuring probability calibration accuracy.
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(md)


def main() -> None:
    """CLI entrypoint for model evaluation."""
    parser = argparse.ArgumentParser(
        description="ClaimRisk AI - Model Evaluation & Threshold Analysis"
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=PROJECT_ROOT / "models",
        help="Directory containing trained model artifacts (.joblib)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing preprocessed split CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "models",
        help="Directory to save evaluation reports and plots",
    )

    args = parser.parse_args()
    run_comprehensive_evaluation(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
