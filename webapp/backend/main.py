"""
main.py — FastAPI application for the Accessible Caption Assistant.

Endpoints:
    GET  /health   → system readiness check
    POST /predict  → upload image, receive caption + confidence + route
    GET  /policy   → current reroute threshold and policy metadata
"""

import csv
import json as _json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import inference
import download_artifacts

# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------
_ART = config.PROJECT_ROOT / "artifacts"


def _safe_read_json(path: Path) -> dict:
    try:
        if path.exists():
            return _json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _safe_read_csv(path: Path, max_rows: int = 100) -> list[dict]:
    try:
        if not path.exists():
            return []
        rows: list[dict] = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if i >= max_rows:
                    break
                rows.append(dict(row))
        return rows
    except Exception:
        return []

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Accessible Caption Assistant API",
    description=(
        "Generates image captions using a trained CNN+LSTM model. "
        "Each prediction includes a confidence score and a route decision "
        "(AUTO or HUMAN_REVIEW) based on the deployment policy threshold."
    ),
    version="1.0.0",
)

# CORS — allow the React frontend to call us from localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    demo_mode: bool
    warnings: list[str]
    rag_available: bool = False

class PredictResponse(BaseModel):
    request_id: str
    caption: str
    confidence: float
    route: str
    rationale: str
    decode_method: str
    demo_mode: bool
    timestamp: str
    rag_caption: str | None = None
    rag_used: bool = False
    retrieved_captions: list[str] = []
    retrieval_sim: float | None = None
    rag_available: bool = False

class PolicyResponse(BaseModel):
    threshold: float
    beam_size: int
    length_penalty: float
    max_len: int
    policy_metadata: dict

# ---------------------------------------------------------------------------
# Startup — load models once
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("  Accessible Caption Assistant — Starting up")
    print("=" * 60)

    # Download model + tokenizer from HF Hub if not present locally
    try:
        model_path, tokenizer_path = download_artifacts.ensure_artifacts()
        config.MODEL_PATH = model_path
        config.TOKENIZER_PATH = tokenizer_path
    except Exception as e:
        print(f"  ❌ Artifact download failed: {e}")

    # Validate config
    warnings = config.validate()
    for w in warnings:
        print(f"  ⚠️  {w}")

    # Load models
    result = inference.load_models()
    if result["ok"] and not result.get("demo_mode"):
        print("  ✅ All models loaded successfully")
    elif result["ok"] and result.get("demo_mode"):
        print("  ⚡ DEMO MODE — returning sample COCO captions (TF not required)")
        for err in result.get("errors", []):
            print(f"     ↳ {err}")
    else:
        for err in result["errors"]:
            print(f"  ❌ {err}")
        print("  ⚠️  Server starting in DEGRADED mode — /predict will return errors")

    # Ensure log file exists with header
    _init_log_file()

    print(f"  🔧 Reroute threshold: {config.REROUTE_THRESHOLD:.4f}")
    print(f"  📁 Project root: {config.PROJECT_ROOT}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Prediction logging
# ---------------------------------------------------------------------------
def _init_log_file():
    """Create prediction log CSV with header if it doesn't exist."""
    log_path = config.LOG_PATH
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_id", "timestamp", "filename",
                "caption", "confidence", "route",
                "decode_method", "log_prob",
            ])


def _log_prediction(
    request_id: str,
    filename: str,
    result: inference.CaptionResult,
    decode_method: str,
):
    """Append one row to the prediction log."""
    with open(config.LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            request_id,
            datetime.now(timezone.utc).isoformat(),
            filename,
            result.caption,
            result.confidence,
            result.route,
            decode_method,
            result.log_prob,
        ])


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """System readiness check."""
    warnings = config.validate()
    status = "ok" if inference.is_ready() and not inference.is_demo() else "demo" if inference.is_ready() else "degraded"
    return HealthResponse(
        status=status,
        models_loaded=inference.is_ready(),
        demo_mode=inference.is_demo(),
        warnings=warnings,
        rag_available=inference.is_rag_ready(),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), use_rag: bool = False):
    """
    Accept an uploaded image and return a generated caption
    with confidence score and route decision.
    Pass use_rag=true to enable retrieval-augmented refinement.
    """
    # --- Validate input ---------------------------------------------------
    if not inference.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Check server logs and /health endpoint.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got content-type '{file.content_type}'.",
        )

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {e}")

    if len(image_bytes) < 100:
        raise HTTPException(status_code=400, detail="Uploaded file appears to be empty or corrupt.")

    # --- Run inference ----------------------------------------------------
    request_id = str(uuid.uuid4())[:8]
    decode_method = f"beam_{config.BEAM_SIZE}_lp{config.LENGTH_PENALTY}"

    try:
        result = inference.generate_caption(image_bytes, use_beam=True, use_rag=use_rag)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    # --- Log to CSV -------------------------------------------------------
    try:
        _log_prediction(request_id, file.filename or "unknown", result, decode_method)
    except Exception:
        pass  # Logging should never crash the request

    # --- Return response --------------------------------------------------
    return PredictResponse(
        request_id=request_id,
        caption=result.caption,
        confidence=result.confidence,
        route=result.route,
        rationale=result.rationale,
        decode_method="demo" if result.demo_mode else decode_method,
        demo_mode=result.demo_mode,
        timestamp=datetime.now(timezone.utc).isoformat(),
        rag_caption=result.rag_caption,
        rag_used=result.rag_used,
        retrieved_captions=result.retrieved_captions,
        retrieval_sim=result.retrieval_sim,
        rag_available=inference.is_rag_ready(),
    )


@app.get("/policy", response_model=PolicyResponse)
async def policy():
    """Return the current deployment policy configuration."""
    return PolicyResponse(
        threshold=config.REROUTE_THRESHOLD,
        beam_size=config.BEAM_SIZE,
        length_penalty=config.LENGTH_PENALTY,
        max_len=config.MAX_LEN,
        policy_metadata=config.POLICY_META,
    )


# ---------------------------------------------------------------------------
# Metrics + analysis endpoints
# ---------------------------------------------------------------------------

class MetricsSummaryResponse(BaseModel):
    stages: list[dict[str, Any]]
    decode_ablation: list[dict[str, Any]]
    train_ablation: list[dict[str, Any]]
    available: bool


class StageMetricsResponse(BaseModel):
    stage: str
    metrics: dict[str, Any]
    sample_predictions: list[dict[str, Any]]
    available: bool


class ErrorAnalysisResponse(BaseModel):
    overall: dict[str, Any]
    error_buckets: list[dict[str, Any]]
    route_summary: list[dict[str, Any]]
    rag_usage: list[dict[str, Any]]
    top_best: list[dict[str, Any]]
    top_worst: list[dict[str, Any]]
    top_rag_improved: list[dict[str, Any]]
    top_rag_degraded: list[dict[str, Any]]
    available: bool


class RerouteResponse(BaseModel):
    sweep: list[dict[str, Any]]
    selected: dict[str, Any]
    available: bool


@app.get("/metrics/summary", response_model=MetricsSummaryResponse)
async def metrics_summary():
    """Stage-by-stage BLEU progression + ablation tables."""
    stages = _safe_read_csv(_ART / "final" / "final_metrics_summary.csv")
    decode = _safe_read_csv(_ART / "stage8_ablation" / "tables" / "decode_ablation.csv")
    train = _safe_read_csv(_ART / "stage8_ablation" / "tables" / "train_optimizer_ablation.csv")
    return MetricsSummaryResponse(
        stages=stages,
        decode_ablation=decode,
        train_ablation=train,
        available=bool(stages),
    )


@app.get("/metrics/stage/{stage}", response_model=StageMetricsResponse)
async def metrics_stage(stage: str):
    """Per-stage metrics JSON + sample caption predictions."""
    allowed = {"stage3", "stage4", "stage5", "stage6", "stage7"}
    if stage not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown stage. Must be one of: {sorted(allowed)}")
    metrics = _safe_read_json(_ART / stage / "metrics.json")
    preds = _safe_read_csv(_ART / stage / "sample_caption_predictions.csv", max_rows=10)
    return StageMetricsResponse(
        stage=stage,
        metrics=metrics,
        sample_predictions=preds,
        available=bool(metrics),
    )


@app.get("/analysis/error", response_model=ErrorAnalysisResponse)
async def analysis_error():
    """Error buckets, route quality, and top-N caption examples."""
    ea = _ART / "final" / "error_analysis"
    overall_rows = _safe_read_csv(ea / "overall_summary.csv")
    return ErrorAnalysisResponse(
        overall=overall_rows[0] if overall_rows else {},
        error_buckets=_safe_read_csv(ea / "error_bucket_counts.csv"),
        route_summary=_safe_read_csv(ea / "route_summary.csv"),
        rag_usage=_safe_read_csv(ea / "rag_usage_summary.csv"),
        top_best=_safe_read_csv(ea / "top20_best_base_bleu4.csv", max_rows=20),
        top_worst=_safe_read_csv(ea / "top20_worst_base_bleu4.csv", max_rows=20),
        top_rag_improved=_safe_read_csv(ea / "top20_rag_improved.csv", max_rows=20),
        top_rag_degraded=_safe_read_csv(ea / "top20_rag_degraded.csv", max_rows=20),
        available=bool(overall_rows),
    )


@app.get("/policy/reroute", response_model=RerouteResponse)
async def policy_reroute():
    """Reroute threshold sweep data + selected deployment policy."""
    sweep = _safe_read_csv(_ART / "stage6" / "reroute_threshold_sweep.csv")
    selected = _safe_read_json(_ART / "final" / "deployment_policy.json")
    return RerouteResponse(
        sweep=sweep,
        selected=selected,
        available=bool(sweep),
    )
