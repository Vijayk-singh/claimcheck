# ClaimRisk AI – Explainable Healthcare Claim Risk Prediction & Analyst Copilot

ClaimRisk AI is a decision-support and audit-prioritization system for healthcare insurance billing claims. It blends synthetic billing dataset generation, statistical anomaly detection, supervised machine learning triage, Explainable AI (SHAP), and an LLM-assisted claims analyst copilot to streamline manual review workflows.

> **Disclaimer**: ClaimRisk AI is a **decision-support prototype** engineered for review triage and prioritization. It is not an automated denial engine, clinical adjudication system, or real-world fraud conviction mechanism.

---

## 1. Problem Statement

Healthcare payers (insurers, government programs) process billions of billing claims annually. Operational reality presents two major challenges:
1. **Auditor Bottlenecks**: Payers cannot manually audit 100% of claims. Pure random sampling is inefficient, while crude deterministic rules generate unmanageable volumes of false-positive review flags.
2. **The "Black Box" Trust Deficit**: Statistical and deep learning models often output risk scores without actionable clinical or billing context. Reviewers need grounded rationales (e.g. *procedure benchmark fee deviation*, *timely filing friction*, *historical provider rejection spikes*) rather than opaque scores.

**ClaimRisk AI** bridges this gap by scoring claims using multi-factor risk modeling and delivering explainable feature attributions alongside evidence-grounded summary narratives for audit teams.

---

## 2. Planned Architecture

The project is structured into modular phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          ClaimRisk AI Platform                         │
├─────────────────────────┬─────────────────────────┬────────────────────┤
│   Phase 1 & 2: Data     │   Phase 3 & 4: Model    │   Phase 5 & 6: UI  │
│   & Feature Pipelines   │   & Explainability      │   & Copilot        │
├─────────────────────────┼─────────────────────────┼────────────────────┤
│ • Synthetic Data Gen    │ • Baseline & ML Models  │ • FastAPI Backend  │
│ • Anomaly Injection     │ • Cost-sensitive tuning │ • React/Vite UI    │
│ • Schema Validation     │ • SHAP Interpretability │ • Recharts Charts  │
│ • Preprocessing & EDA   │ • Risk Tier Assignment  │ • Copilot Dialogue │
└─────────────────────────┴─────────────────────────┴────────────────────┘
```

### End-to-End Pipeline
1. **Synthetic Claim Generation**: Probabilistic generator embedding standard procedure benchmarks (CPT/HCPCS proxies), realistic provider behavioral tiers, and realistic noise.
2. **Feature Engineering**: Standardizes charges, calculates z-score fee deviations against clinical code benchmarks, and captures provider filing history.
3. **ML Risk Classifier**: Calibrated classification predicting whether a claim warrants manual audit (`requires_review` = 0 or 1).
4. **Explainability Engine (SHAP)**: Extracts local feature contributions explaining why a particular claim was selected for review.
5. **Analyst Copilot & UI**: Presents claims queue, drill-down panels, feature impact charts, and natural language audit summaries.

---

## 3. Dataset Schema

The Stage 1 dataset generates 20,000 healthcare billing claims with 14 standardized features:

| Field Name | Type | Description | Operational Scope |
|:---|:---|:---|:---|
| `claim_id` | String | Unique claim identifier (`CLM-<seed>-XXXXXX`) | Primary Key |
| `provider_id` | String | Healthcare provider identifier (`PRV-XXXX`) | Categorical Identifier |
| `procedure_code` | String | CPT/HCPCS standard procedure code proxy | Clinical Feature |
| `claim_amount` | Float | Total dollar charge billed by the provider | Pre-adjudication Feature |
| `approved_amount` | Float | Contractual allowable amount determined post-review | **Post-adjudication Outcome** |
| `service_date` | Date (YYYY-MM-DD) | Date healthcare service was administered | Pre-adjudication Feature |
| `submission_date` | Date (YYYY-MM-DD) | Date claim was submitted to payer | Pre-adjudication Feature |
| `submission_delay` | Integer | Elapsed days between service and submission | Timely Filing Indicator |
| `missing_fields` | Integer | Count of non-critical missing billing fields (0 to 4) | Documentation Friction |
| `duplicate_indicator` | Binary (0 / 1) | Re-submission flag indicating suspected duplicate billing | Risk Indicator |
| `provider_claim_frequency`| Integer | Monthly average claim volume submitted by provider | Historical Provider Metric |
| `historical_rejection_rate`| Float | Rolling historical rejection percentage (0.01 - 0.65) | Historical Provider Metric |
| `amount_deviation` | Float | Z-score deviation from standard procedure benchmark fee | Pricing Anomaly Metric |
| `requires_review` | Binary (0 / 1) | **Target Variable**: 1 = Requires Manual Audit, 0 = Clean / Pass | Supervised ML Target |

> **Crucial Data Hygiene Note**: `approved_amount` is a simulated post-adjudication outcome (established only *after* review and adjudication). It is included for end-to-end lifecycle simulation, but is strictly isolated from the training feature set to prevent data leakage.

---

## 4. Synthetic Data Generation Methodology

The generation engine in [`backend/ml/generate_data.py`](file:///home/vijay/Desktop/clainrisk/claimrisk-ai/backend/ml/generate_data.py) implements realistic stochastic distributions rather than trivial hardcoded rules:

1. **Provider Profiles**:
   - 80% Standard Compliant Providers (low rejections, normal cadence).
   - 12% High-Volume Facilities (high throughput, moderate clerical rejections).
   - 8% Elevated Risk Providers (higher rejection history, burst submissions).
2. **Procedure Catalog Benchmarking**:
   - Encodes representative outpatient, emergency, radiology, surgical, and laboratory codes with distinct Gaussian fee benchmarks ($\mu, \sigma$) and realistic billing frequencies.
3. **Controlled Anomaly Injection**:
   - Upcoding / fee skewing (~6% frequency, multiplier 2.2x - 5.5x base fee).
   - Filing delays (Gamma distribution with a heavy tail simulating timely filing issues).
   - Duplicate submissions (correlated with provider risk tier).
4. **Probabilistic Review Target Assignment**:
   The target `requires_review` is generated via a latent logistic score model:
   $$\text{logit} = -2.20 + 0.90 \cdot z_{\text{amt}} + 1.85 \cdot z_{\text{dup}} + 0.80 \cdot z_{\text{rej}} + 0.45 \cdot z_{\text{delay}} + 0.55 \cdot z_{\text{missing}} + 0.35 \cdot (z_{\text{amt}} \times z_{\text{rej}}) + 0.50 \cdot I_{\text{elevated}} + \epsilon$$
   where $\epsilon \sim \mathcal{N}(0, 0.45^2)$ injects realistic auditor variance and unobserved administrative factors.
   - Resulting dataset review rate: **~28.8% positive review rate**.
   - No single feature deterministically separates the classes.

---

## 5. Project Directory Structure

```
claimrisk-ai/
├── backend/
│   ├── app/                       # FastAPI application (Phase 5)
│   ├── ml/                        # ML modules & pipeline logic
│   │   ├── __init__.py
│   │   ├── generate_data.py       # Synthetic dataset generator & schema validator
│   │   ├── validate_data.py       # Standalone dataset CLI validator
│   │   ├── eda.py                 # Exploratory data analysis & reporting (Phase 2)
│   │   ├── feature_engineering.py # Domain feature transformations (Phase 2)
│   │   ├── preprocessing.py       # ML-ready preprocessing pipeline & splits (Phase 2)
│   │   ├── train.py               # Model training & tuning (Phase 3)
│   │   ├── evaluate.py            # Evaluation metrics & reports (Phase 3)
│   │   └── predict.py             # Batch & real-time inference (Phase 4)
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_generate_data.py        # Data generator & schema tests (15 tests)
│   │   ├── test_feature_engineering.py  # Feature engineering & leakage tests (11 tests)
│   │   ├── test_preprocessing.py        # Preprocessing, encoding & split tests (11 tests)
│   │   └── test_eda.py                  # Profiling, quality & visualization tests (7 tests)
│   └── requirements.txt           # Pinned project dependencies
├── data/
│   ├── synthetic/
│   │   └── healthcare_claims.csv  # Generated 20,000-row synthetic dataset
│   └── processed/                 # Partitioned splits & fitted preprocessor artifacts (Phase 2)
│       ├── X_train_processed.csv
│       ├── X_test_processed.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── feature_names.json
│       └── preprocessor.joblib
├── reports/
│   └── eda/                       # Phase 2 EDA reports & visualizations
│       ├── eda_summary.json
│       ├── eda_report.md
│       ├── target_distribution.png
│       ├── claim_amount_distribution.png
│       ├── submission_delay_distribution.png
│       ├── amount_deviation_distribution.png
│       ├── missing_fields_distribution.png
│       ├── numeric_correlation_heatmap.png
│       ├── features_by_target.png
│       └── risk_indicators_by_target.png
├── docs/                          # Architecture diagrams & documentation
├── notebooks/                     # Interactive Jupyter exploration
├── frontend/                      # React + Vite dashboard (Phase 6)
├── .gitignore
└── README.md
```

---

## 6. Execution Guide & CLI Commands

### Prerequisites
- Linux OS (Ubuntu 20.04+ or compatible)
- Python 3.10+ (Tested on Python 3.10 – 3.13)

### Step 1: Environment Setup
```bash
cd claimrisk-ai

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Step 2: Generate Synthetic Dataset (Phase 1)
To generate the default 20,000 claims with seed 42:
```bash
python backend/ml/generate_data.py --size 20000 --seed 42 --output data/synthetic/healthcare_claims.csv
```

To validate dataset schema, bounds, and business rules:
```bash
python backend/ml/validate_data.py --input data/synthetic/healthcare_claims.csv
```

### Step 3: Run Exploratory Data Analysis & Generate Visualizations (Phase 2)
To profile the dataset, perform quality auditing, generate all 8 high-resolution Matplotlib visualizations, and produce JSON/Markdown audit reports:
```bash
python backend/ml/eda.py --input data/synthetic/healthcare_claims.csv --output-dir reports/eda
```

Output deliverables generated in `reports/eda/`:
- `eda_summary.json`: Comprehensive structured profiling and anomaly audit statistics.
- `eda_report.md`: Executive analysis detailing observations, class balance, outliers, and recommendations.
- 8 high-resolution visualization PNGs:
  1. `target_distribution.png`
  2. `claim_amount_distribution.png`
  3. `submission_delay_distribution.png`
  4. `amount_deviation_distribution.png`
  5. `missing_fields_distribution.png`
  6. `numeric_correlation_heatmap.png`
  7. `features_by_target.png`
  8. `risk_indicators_by_target.png`

### Step 4: Run Feature Engineering (Phase 2)
To enrich claims with domain indicators, benchmark fee ratios, and temporal date features:
```bash
python backend/ml/feature_engineering.py --input data/synthetic/healthcare_claims.csv --output data/processed/claims_engineered.csv
```

Engineered features:
- `amount_to_benchmark_ratio`: Relative procedure fee markup normalized against procedure code baseline mean.
- `high_submission_delay`: Binary indicator for delayed billing filing (> 60 days).
- `has_multiple_missing_fields`: Binary indicator for clerical documentation deficits (missing fields >= 2).
- `high_provider_rejection_rate`: Binary indicator for elevated provider rejection history (> 0.20).
- `high_provider_claim_frequency`: Binary indicator for high-throughput billing facilities (> 300 claims/month).
- `high_amount_deviation`: Pricing anomaly flag for charges exceeding 2 standard deviations (> 2.0 Z).
- `service_month`: Calendar month of clinical service (1-12).
- `service_day_of_week`: Day of week of clinical service (0-6).
- `service_is_weekend`: Binary flag for weekend elective service encounters.

### Step 5: Execute Preprocessing Pipeline & Partitioning (Phase 2)
To isolate predictive features, strictly drop post-adjudication outcomes (`approved_amount`) and identifiers (`claim_id`, `provider_id`), perform an 80:20 stratified split, and fit transformers solely on training data:
```bash
python backend/ml/preprocessing.py --input data/synthetic/healthcare_claims.csv --output-dir data/processed --test-size 0.20 --seed 42
```

Outputs saved to `data/processed/`:
- `X_train_processed.csv`: Scaled and one-hot encoded training feature matrix (16,000 records × 28 features).
- `X_test_processed.csv`: Transformed testing feature matrix (4,000 records × 28 features).
- `y_train.csv` / `y_test.csv`: Stratified target partitions (~28.86% positive review rate).
- `X_train_raw.csv` / `X_test_raw.csv`: Unscaled feature splits with engineered variables.
- `feature_names.json`: Ordered list of 28 model-ready input feature names.
- `preprocessor.joblib`: Serialized ColumnTransformer fitted strictly on training data.

### Step 6: Train & Compare ML Classifiers & PyTorch Baseline (Phase 3)
To train and benchmark candidate models across scikit-learn (Logistic Regression, Random Forest, HistGradientBoosting) and a deep neural network PyTorch baseline (`ClaimRiskNN`) with Stratified Cross-Validation:
```bash
python backend/ml/train.py --data-dir data/processed --output-dir models
```

Outputs generated in `models/`:
- `LogisticRegression.joblib`: Linear baseline classifier with balanced class weighting.
- `RandomForest.joblib`: 150-tree bagging ensemble.
- `HistGradientBoosting.joblib`: Gradient-boosted decision trees with early stopping.
- `PyTorch_NN.pt` & `PyTorch_NN.joblib`: Deep PyTorch neural network checkpoint (Batch Normalization, Dropout, BCEWithLogitsLoss).
- `champion_model.joblib`: Automatically selected top-performing model ranked by PR-AUC.
- `model_comparison.json`: Structured CV and test evaluation benchmark results.

### Step 7: Run Model Evaluation & Threshold Analysis (Phase 3)
To execute the comprehensive evaluation pipeline, perform decision threshold sweep (0.05 to 0.95), simulate operational audit costs, and generate all publication visualizations:
```bash
python backend/ml/evaluate.py --models-dir models --data-dir data/processed --output-dir reports/models
```

Visualizations and reports generated in `reports/models/`:
- `model_comparison_roc.png`: Multi-model ROC curves with AUC ratings.
- `model_comparison_pr.png`: Precision-Recall curves (critical for imbalanced audit data).
- `threshold_analysis.png`: Precision, Recall, F1, and Review Workload vs decision threshold.
- `confusion_matrices.png`: Confusion matrices at operating threshold.
- `calibration_curves.png`: Reliability diagram measuring probability calibration.
- `evaluation_summary.json` & `evaluation_report.md`: Executive audit evaluation report.

### Step 8: Run Explainable AI Module (Phase 4)
To compute global SHAP feature importance rankings and export the serializable explainer:
```bash
python backend/ml/explain.py --model-path models/champion_model.joblib --data-dir data/processed --output-dir reports/explainability
```

Deliverables generated in `reports/explainability/`:
- `shap_feature_importance.png`: Top-15 global feature importance bar chart.
- `explainability_summary.json`: Ranked feature contribution scores.
- `explainer.joblib`: Pre-configured explainer for sub-millisecond API inference.

### Step 9: Launch FastAPI Backend & Analyst Copilot Dashboard (Phase 5 & 6)
To start the production-ready FastAPI service:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Web Dashboard**: Open [http://localhost:8000](http://localhost:8000) or `/dashboard` in your browser.
- **Interactive Swagger API Docs**: Open [http://localhost:8000/docs](http://localhost:8000/docs).

### Step 10: Run All Automated Tests
Execute the comprehensive test suite (65 unit and integration tests):
```bash
pytest backend/tests/ -v
```

---

## 7. Model Performance Benchmark Comparison

Evaluated on a strictly isolated 4,000-claim held-out test partition:

| Model Candidate | PR-AUC (Priority) | ROC-AUC | F1-Score (0.50) | Test Recall | Test Precision | Architecture / Estimator |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **HistGradientBoosting 🏆** | **0.7526** | **0.8486** | **0.6502** | 72.79% | 58.74% | Gradient Boosted Trees (Champion) |
| **LogisticRegression** | 0.7517 | 0.8502 | 0.6504 | 70.54% | 60.34% | Calibrated L2 Linear Baseline |
| **PyTorch Neural Network** | 0.7506 | 0.8500 | 0.6548 | 72.01% | 60.04% | Deep MLP (BatchNorm + Dropout) |
| **RandomForest** | 0.7391 | 0.8404 | 0.6370 | 64.47% | 62.94% | 150-Tree Bagging Ensemble |

---

## 8. Decision Threshold & Review Queue Optimization

Threshold analysis enables healthcare audit managers to tune review policies to their operational staffing capacity:

| Operating Policy | Decision Threshold | Audit Recall (Risk Capture) | Audit Precision | Review Workload (% Queue) | Simulated Cost |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Aggressive Audit (High Recall)** | `0.20` | **93.8%** | 42.1% | 64.3% of claims | $86,400 |
| **Cost-Optimal Policy** | `0.35` | **83.6%** | 52.8% | 45.7% of claims | **$66,900** |
| **Balanced Default** | `0.50` | **72.8%** | 58.7% | 35.8% of claims | $68,250 |
| **Lean Audit (High Precision)** | `0.70` | **48.2%** | **78.4%** | 17.7% of claims | $79,500 |

---

## 9. Current Status & Deliverables

| Deliverable | Phase | Status | Details |
|:---|:---:|:---:|:---|
| **Synthetic Claims Generator** | 1 | Completed | 20,000 records at `data/synthetic/healthcare_claims.csv` |
| **Schema Validation CLI** | 1 | Completed | Zero-null, unique key, and domain constraint checks |
| **Exploratory Data Analysis** | 2 | Completed | Reusable `backend/ml/eda.py` module & CLI |
| **Data Quality & Audit Reports** | 2 | Completed | Generated `reports/eda/eda_summary.json` & `eda_report.md` |
| **EDA Visualizations** | 2 | Completed | 8 publication-grade Matplotlib PNGs in `reports/eda/` |
| **Feature Engineering Pipeline** | 2 | Completed | 9 explainable domain features in `backend/ml/feature_engineering.py` |
| **ML Preprocessing & Splits** | 2 | Completed | Stratified 80:20 split & ColumnTransformer in `backend/ml/preprocessing.py` |
| **Model Comparison Workflow** | 3 | Completed | Scikit-learn models & PyTorch baseline (`backend/ml/train.py`) |
| **Evaluation & Threshold Analysis** | 3 | Completed | PR-AUC, ROC-AUC, threshold curves (`backend/ml/evaluate.py`) |
| **Explainable AI (SHAP)** | 4 | Completed | Local & global feature attribution (`backend/ml/explain.py`) |
| **Analyst Copilot Engine** | 5 | Completed | Grounded LLM review memos & Q&A (`backend/ml/copilot.py`) |
| **FastAPI REST API Service** | 5 | Completed | REST endpoints with OpenAPI documentation (`backend/app/main.py`) |
| **Analyst Web Dashboard** | 6 | Completed | Interactive UI with queue, SHAP waterfall, chat (`frontend/index.html`) |
| **Automated Test Suite** | 1 - 6 | Completed | **65 automated tests passing** across all modules |
