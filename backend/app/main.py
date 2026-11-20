"""
ClaimRisk AI - FastAPI Application & Analyst Copilot Service.

Exposes REST APIs for healthcare claims risk triage, explainable AI, and copilot dialogue:
- POST /api/predict: Real-time claim scoring, risk tier assignment, and SHAP attribution.
- POST /api/predict/batch: Batch review triage and financial exposure queue prioritization.
- GET /api/claims: Query claims repository with multi-attribute filtering and pagination.
- GET /api/claims/{claim_id}: Deep-dive claim profile with local SHAP waterfall.
- POST /api/copilot/summary: Grounded LLM audit memo synthesis.
- POST /api/copilot/chat: Interactive copilot dialogue for auditor inquiries.
- GET /api/models/comparison: Benchmarks across scikit-learn models & PyTorch baseline.
- GET /api/explain/global: Global feature importance rankings.
- GET /: Embedded Analyst Copilot Web Dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from backend.ml.copilot import PROCEDURE_BENCHMARK_MAP, ClaimRiskCopilot
from backend.ml.predict import ClaimRiskScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claimrisk.api")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models, copilot, and dataset caches on application startup."""
    global scorer, copilot, claims_cache
    logger.info("Initializing ClaimRisk AI services...")

    try:
        scorer = ClaimRiskScorer()
        logger.info("ClaimRiskScorer initialized with champion model.")
    except Exception as e:
        logger.warning(f"Could not initialize scorer on startup: {e}")

    try:
        copilot = ClaimRiskCopilot()
        logger.info("ClaimRiskCopilot initialized.")
    except Exception as e:
        logger.warning(f"Could not initialize copilot on startup: {e}")

    # Load claims cache if synthetic data exists
    claims_csv = PROJECT_ROOT / "data" / "synthetic" / "healthcare_claims.csv"
    if claims_csv.exists():
        try:
            claims_cache = pd.read_csv(claims_csv)
            logger.info(f"Loaded {len(claims_cache)} claims into memory cache.")
        except Exception as e:
            logger.warning(f"Failed to load claims cache: {e}")
    yield


# Initialize FastAPI App
app = FastAPI(
    title="ClaimRisk AI – Healthcare Claim Risk Prediction & Analyst Copilot",
    description="Explainable Decision-Support & Review Prioritization for Healthcare Billing Audits",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for cross-origin client apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
scorer: Optional[ClaimRiskScorer] = None
copilot: Optional[ClaimRiskCopilot] = None
claims_cache: Optional[pd.DataFrame] = None


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class ClaimInput(BaseModel):
    """Input schema representing a single healthcare billing claim."""
    claim_id: Optional[str] = Field("CLM-DEMO-001", description="Unique claim identifier")
    provider_id: Optional[str] = Field("PRV-1001", description="Billing provider ID")
    procedure_code: str = Field("99214", description="CPT/HCPCS procedure code")
    claim_amount: float = Field(250.0, ge=0.0, description="Total billed charge ($)")
    service_date: Optional[str] = Field("2024-03-01", description="Date of clinical service (YYYY-MM-DD)")
    submission_date: Optional[str] = Field("2024-03-15", description="Date of claim submission (YYYY-MM-DD)")
    submission_delay: int = Field(14, ge=0, description="Elapsed days between service and submission")
    missing_fields: int = Field(0, ge=0, le=10, description="Number of omitted non-critical billing fields")
    duplicate_indicator: int = Field(0, ge=0, le=1, description="Binary suspected duplicate submission flag")
    provider_claim_frequency: int = Field(150, ge=0, description="Monthly claim volume of provider")
    historical_rejection_rate: float = Field(0.08, ge=0.0, le=1.0, description="Historical claim rejection rate")
    amount_deviation: float = Field(0.20, description="Standardized benchmark fee deviation Z-score")


class FeatureAttributionItem(BaseModel):
    feature: str
    display_name: str
    value: float
    shap_value: float
    impact: str


class ExplanationResponse(BaseModel):
    predicted_risk_probability: float
    top_risk_drivers: List[FeatureAttributionItem]
    top_mitigating_factors: List[FeatureAttributionItem]
    all_attributions: List[FeatureAttributionItem]


class ClaimPredictionResponse(BaseModel):
    claim_id: str
    provider_id: str
    procedure_code: str
    claim_amount: float
    risk_score: float
    risk_tier: str
    requires_review: int
    expected_financial_exposure: float
    recommended_action: str
    explanation: Optional[ExplanationResponse] = None


class BatchPredictionRequest(BaseModel):
    claims: List[ClaimInput]
    prioritize_by: str = Field("risk_score", description="Sort queue by 'risk_score' or 'financial_exposure'")


class CopilotSummaryRequest(BaseModel):
    claim: ClaimInput
    prediction: Optional[Dict[str, Any]] = None


class CopilotChatRequest(BaseModel):
    question: str
    claim: ClaimInput
    prediction: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/api/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Return health status and model availability."""
    return {
        "status": "healthy",
        "service": "ClaimRisk AI API",
        "scorer_loaded": scorer is not None,
        "copilot_loaded": copilot is not None,
        "claims_loaded": len(claims_cache) if claims_cache is not None else 0,
        "version": "1.0.0",
    }


@app.post("/api/predict", response_model=ClaimPredictionResponse, tags=["Scoring & Triage"])
def predict_single_claim(claim: ClaimInput) -> Dict[str, Any]:
    """Score a single claim, assign risk tier, and return local SHAP explanations."""
    global scorer
    if scorer is None:
        scorer = ClaimRiskScorer()

    claim_dict = claim.model_dump()
    result = scorer.predict_single(claim_dict, include_explanation=True)
    return result


@app.post("/api/predict/batch", tags=["Scoring & Triage"])
def predict_batch_claims(batch_req: BatchPredictionRequest) -> Dict[str, Any]:
    """Batch score multiple claims and return prioritized review queue."""
    global scorer
    if scorer is None:
        scorer = ClaimRiskScorer()

    claims_data = [c.model_dump() for c in batch_req.claims]
    df = pd.DataFrame(claims_data)
    scored_df = scorer.predict_batch(df, prioritize_by=batch_req.prioritize_by)

    return {
        "count": len(scored_df),
        "prioritized_by": batch_req.prioritize_by,
        "claims": scored_df.to_dict(orient="records"),
    }


@app.get("/api/claims", tags=["Claims Database"])
def list_claims(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    risk_tier: Optional[str] = Query(None, description="Filter: LOW, MEDIUM, HIGH"),
    procedure_code: Optional[str] = Query(None, description="Filter by CPT code"),
    duplicate_only: bool = Query(False, description="Filter only duplicate-flagged claims"),
    search: Optional[str] = Query(None, description="Search claim_id or provider_id"),
) -> Dict[str, Any]:
    """Browse stored claims with filtering, pagination, and real-time risk scores."""
    global claims_cache, scorer
    if claims_cache is None:
        claims_csv = PROJECT_ROOT / "data" / "synthetic" / "healthcare_claims.csv"
        if claims_csv.exists():
            claims_cache = pd.read_csv(claims_csv)
        else:
            raise HTTPException(status_code=404, detail="Synthetic claims dataset not found. Generate data first.")

    df = claims_cache.copy()

    # Pre-score top records if not scored
    if "risk_score" not in df.columns and scorer is not None:
        scored = scorer.predict_batch(df.head(2000))
        df.loc[scored.index, "risk_score"] = scored["risk_score"]
        df.loc[scored.index, "risk_tier"] = scored["risk_tier"]
        df.loc[scored.index, "financial_exposure"] = scored["financial_exposure"]

    # Apply filters
    if procedure_code:
        df = df[df["procedure_code"] == procedure_code]
    if duplicate_only:
        df = df[df["duplicate_indicator"] == 1]
    if search:
        search_str = search.strip().upper()
        df = df[
            df["claim_id"].str.upper().str.contains(search_str, na=False)
            | df["provider_id"].str.upper().str.contains(search_str, na=False)
        ]
    if risk_tier and "risk_tier" in df.columns:
        df = df[df["risk_tier"] == risk_tier.upper()]

    total_records = len(df)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    records = df.iloc[start_idx:end_idx].to_dict(orient="records")

    return {
        "page": page,
        "page_size": page_size,
        "total_records": total_records,
        "total_pages": (total_records + page_size - 1) // page_size,
        "claims": records,
    }


@app.get("/api/claims/{claim_id}", tags=["Claims Database"])
def get_claim_detail(claim_id: str) -> Dict[str, Any]:
    """Retrieve detailed profile, benchmark comparisons, and SHAP waterfall for a single claim."""
    global claims_cache, scorer
    if claims_cache is None:
        claims_csv = PROJECT_ROOT / "data" / "synthetic" / "healthcare_claims.csv"
        if claims_csv.exists():
            claims_cache = pd.read_csv(claims_csv)
        else:
            raise HTTPException(status_code=404, detail="Claims dataset not found.")

    matches = claims_cache[claims_cache["claim_id"] == claim_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found.")

    claim_record = matches.iloc[0].to_dict()

    if scorer is None:
        scorer = ClaimRiskScorer()

    score_res = scorer.predict_single(claim_record, include_explanation=True)
    bench_info = PROCEDURE_BENCHMARK_MAP.get(
        str(claim_record.get("procedure_code")),
        {"description": "Standard Medical Encounter", "mean_fee": 500.0, "std_fee": 100.0},
    )

    return {
        "claim": claim_record,
        "prediction": score_res,
        "benchmark": bench_info,
    }


@app.post("/api/copilot/summary", tags=["Analyst Copilot"])
def generate_copilot_summary(req: CopilotSummaryRequest) -> Dict[str, Any]:
    """Generate grounded LLM audit memo for a claim."""
    global scorer, copilot
    if scorer is None:
        scorer = ClaimRiskScorer()
    if copilot is None:
        copilot = ClaimRiskCopilot()

    claim_dict = req.claim.model_dump()
    prediction = req.prediction or scorer.predict_single(claim_dict, include_explanation=True)

    summary = copilot.generate_grounded_summary(
        claim_data=claim_dict,
        prediction_result=prediction,
    )
    return summary


@app.post("/api/copilot/chat", tags=["Analyst Copilot"])
def copilot_chat(req: CopilotChatRequest) -> Dict[str, Any]:
    """Conversational dialogue with the Analyst Copilot regarding a specific claim."""
    global scorer, copilot
    if scorer is None:
        scorer = ClaimRiskScorer()
    if copilot is None:
        copilot = ClaimRiskCopilot()

    claim_dict = req.claim.model_dump()
    prediction = req.prediction or scorer.predict_single(claim_dict, include_explanation=True)

    answer = copilot.answer_question(
        question=req.question,
        claim_data=claim_dict,
        prediction_result=prediction,
    )

    return {
        "question": req.question,
        "answer": answer,
        "claim_id": claim_dict.get("claim_id"),
        "risk_tier": prediction.get("risk_tier"),
    }


@app.get("/api/models/comparison", tags=["Model Evaluation"])
def get_model_comparison() -> Dict[str, Any]:
    """Return performance benchmark comparison and threshold analysis."""
    eval_json = PROJECT_ROOT / "reports" / "models" / "evaluation_summary.json"
    comp_json = PROJECT_ROOT / "models" / "model_comparison.json"

    if eval_json.exists():
        with open(eval_json, "r", encoding="utf-8") as f:
            return json.load(f)
    elif comp_json.exists():
        with open(comp_json, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise HTTPException(
            status_code=404,
            detail="Evaluation summary not found. Train and evaluate models first.",
        )


@app.get("/api/explain/global", tags=["Explainability"])
def get_global_explainability() -> Dict[str, Any]:
    """Return global feature importance ranking and SHAP summary."""
    exp_json = PROJECT_ROOT / "reports" / "explainability" / "explainability_summary.json"
    if exp_json.exists():
        with open(exp_json, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise HTTPException(
            status_code=404,
            detail="Global explainability report not found. Run explainability module first.",
        )


# ---------------------------------------------------------
# Embedded Analyst Copilot Web Dashboard
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard() -> str:
    """Serve the complete interactive ClaimRisk AI Analyst Copilot Dashboard."""
    dashboard_html_path = PROJECT_ROOT / "frontend" / "index.html"
    if dashboard_html_path.exists():
        with open(dashboard_html_path, "r", encoding="utf-8") as f:
            return f.read()

    return """<!DOCTYPE html>
<html>
<head><title>ClaimRisk AI</title></head>
<body><h1>ClaimRisk AI Backend Active</h1><p>Visit <a href='/docs'>/docs</a> for API documentation.</p></body>
</html>"""
