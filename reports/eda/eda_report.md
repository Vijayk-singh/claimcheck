# ClaimRisk AI – Exploratory Data Analysis & Quality Audit Report

**Generated Date**: Automated Phase 2 Pipeline  
**Dataset Scope**: Healthcare Billing Claims Triage  
**Output Artifacts Directory**: `reports/eda`

---

## 1. Executive Summary & Key Observations

This report delivers a rigorous exploratory audit of the 20,000 synthetic healthcare billing claims dataset.
Rather than merely echoing summary statistics, the following key behavioral, clinical, and operational dynamics have been identified:

1. **Target Triage Balance (~28.86% Positive Audit Rate)**:
   - Clean / Auto-Adjudicated claims (`requires_review = 0`): **14,229** (71.14%)
   - High-Risk / Manual Review claims (`requires_review = 1`): **5,771** (28.86%)
   - Class imbalance ratio is approximately **2.47:1**. This moderate imbalance is highly realistic for healthcare insurance audit workflows and is well-suited for stratified sampling and PR-AUC/F1 optimization in Phase 3 modeling without requiring heavy synthetic oversampling.

2. **Bimodal / Right-Skewed Pricing Structure**:
   - Billed claim amount ranges from **$13.42** up to **$23410.24** with an overall mean of **$336.65** (median: **$138.49**, skewness: **11.99**).
   - High variance reflects procedural heterogeneity (ranging from routine low-cost venipunctures to high-cost inpatient surgical arthroplasty) combined with controlled pricing anomaly injection.

3. **Controlled Pricing Anomaly Injection (6.33% > 3 Z-score)**:
   - The standardized benchmark deviation (`amount_deviation`) reveals **1,266** claims exceeding 3 standard deviations and **1,073** claims exceeding 10 standard deviations (maximum z-score: **36.94**).
   - When cross-referenced against the review target, claims requiring audit have a substantially higher mean deviation, confirming this feature's high predictive power.

4. **Timely Filing Friction (Mixture Distribution)**:
   - Submission delay exhibits a primary gamma distribution peaking at **15 days** (typical billing operational turnaround) and a long tail extending up to **209 days**.
   - Claims with delayed filing (> 60 days) exhibit a 3.5x higher probability of being routed to manual review.

5. **Administrative Friction & Multi-Factor Risk**:
   - Duplicate submissions (2.65% of claims) and missing documentation fields strongly separate clean claims from review candidates.
   - Importantly, non-determinism holds: no single variable provides 100% linear separation, mimicking real-world auditor decision boundaries.

---

## 2. Dataset Profiling Overview

| Metric | Measured Value | Notes |
|:---|:---|:---|
| **Total Claim Records** | 20,000 | Complete sample |
| **Total Features** | 14 | 14 raw columns |
| **Primary Key Uniqueness** | True | `claim_id` unique across all rows |
| **Total Missing Cells** | 0 | 0.0% nulls in schema |
| **Full Duplicate Rows** | 0 | Clean dataset |
| **Unique Billing Providers** | 250 | Provider population |
| **Mean Claims per Provider** | 80.0 | Distribution across tiers |

---

## 3. Data Quality & Integrity Audit

- **Negative Claim Amounts**: **0** (Zero detected; bounds satisfied)
- **Negative Approved Amounts**: **0** (Zero detected; bounds satisfied)
- **Approved Exceeds Billed**: **0** (Contractual allowable satisfied)
- **Chronological Sequencing**: **0** submission dates precede service dates
- **Delay Consistency**: **0** mismatches between date differences and `submission_delay`
- **Data Quality Status**: **CLEAN**

---

## 4. Visualizations Index

The following 8 high-resolution charts were generated and saved to `reports/eda`:

1. `target_distribution.png`: Binary audit class frequencies and proportions.
2. `claim_amount_distribution.png`: Linear vs. log-scaled billed dollar distributions.
3. `submission_delay_distribution.png`: Filing latency histogram with timeliness cutoff lines.
4. `amount_deviation_distribution.png`: Standardized procedure pricing deviations and outlier bounds.
5. `missing_fields_distribution.png`: Spread of clerical missing non-critical documentation fields.
6. `numeric_correlation_heatmap.png`: Pearson inter-feature correlation matrix.
7. `features_by_target.png`: Density distributions of key signals conditioned on `requires_review`.
8. `risk_indicators_by_target.png`: Relative risk indicator prevalence in review vs. clean cohorts.

---

## 5. Strategic Recommendations for Feature Engineering & Preprocessing

Based on these empirical findings:
1. **Exclude Post-Adjudication Outcome**:
   `approved_amount` has strong post-hoc correlation with the review label ($0 payments for denials) and **MUST be strictly dropped** before model training to eliminate leakage.
2. **Standardize Procedure Price Benchmarks**:
   Create `amount_to_benchmark_ratio` to normalize fee variances across medical specialties.
3. **Encode Operational Flags**:
   Binarize critical non-linear risk thresholds: `high_submission_delay` (> 60 days), `has_multiple_missing_fields` (>= 2), and `high_amount_deviation` (> 2.0).
4. **Temporal Disaggregation**:
   Decompose `service_date` into `service_month`, `service_day_of_week`, and `service_is_weekend` to capture operational scheduling anomalies.
5. **Categorical Handling**:
   One-hot encode `procedure_code` with `handle_unknown="ignore"` to safeguard production scoring against novel medical codes.
6. **Stratified Partitioning**:
   Perform 80:20 train/test stratified split to preserve the ~28.9% audit positive rate across all modeling folds.
