"""FastAPI application factory and endpoints for Banking77 intent classification."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch

from src.api.config import settings
from src.api.schemas import HealthResponse, PredictRequest, PredictResponse, TopPrediction
from src.config import PROJECT_ROOT
from src.inference import Banking77Predictor

FRONTEND_DIR = PROJECT_ROOT / "frontend"

logger = logging.getLogger("banking77_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager to load the predictor once at startup and clean up at shutdown."""
    logger.info("Initializing Banking77Predictor from: %s", settings.model_dir)
    try:
        predictor = Banking77Predictor(
            model_dir=settings.model_dir,
            device=settings.device,
            max_length=settings.max_length,
        )
        app.state.predictor = predictor
        logger.info(
            "Predictor loaded successfully on device '%s' (Init duration: %.2f ms)",
            predictor.device,
            predictor.initialization_time_ms,
        )
    except Exception as exc:
        logger.error("Failed to load predictor checkpoint: %s", exc)
        app.state.predictor = None
        raise exc

    yield

    logger.info("Shutting down Banking77 API service...")
    if hasattr(app.state, "predictor") and app.state.predictor is not None:
        del app.state.predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance with CORS and static UI hosting."""
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=settings.api_description,
        lifespan=lifespan,
    )

    # Enable CORS for frontend flexibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="Service Health Check",
        tags=["System"],
    )
    async def health_check(request: Request) -> HealthResponse:
        """Return the health status of the API service, underlying model, and device telemetry."""
        predictor: Banking77Predictor = getattr(request.app.state, "predictor", None)
        if predictor is None or predictor.model is None:
            return HealthResponse(
                status="unhealthy",
                model_name="distilbert-base-uncased (Banking77 V2)",
                device="unknown",
                model_loaded=False,
            )

        return HealthResponse(
            status="healthy",
            model_name="distilbert-base-uncased (Banking77 V2)",
            device=str(predictor.device),
            model_loaded=True,
        )

    @app.post(
        "/predict",
        response_model=PredictResponse,
        summary="Predict Banking Intent",
        tags=["Inference"],
    )
    async def predict_intent(payload: PredictRequest, request: Request) -> PredictResponse:
        """Classify a customer query into one of the 77 Banking77 intent categories."""
        predictor: Banking77Predictor = getattr(request.app.state, "predictor", None)
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded or service is initializing.",
            )

        # Run inference using the preloaded predictor
        out = predictor.predict(query=payload.text, top_k=3)

        top_candidates = [
            TopPrediction(intent=c.intent_name, probability=round(c.probability, 4))
            for c in out.top_k_predictions
        ]

        return PredictResponse(
            text=payload.text,
            predicted_intent=out.predicted_intent,
            confidence=round(out.confidence, 4),
            top_predictions=top_candidates,
            latency_ms=round(out.latency_ms, 2),
        )

    @app.get(
        "/api",
        summary="API Info",
        include_in_schema=False,
    )
    async def api_info():
        """API metadata endpoint."""
        return JSONResponse(
            content={
                "message": "Banking77 Intent Classification API",
                "docs_url": "/docs",
                "health_url": "/health",
                "predict_url": "/predict",
            }
        )

    # Mount frontend static files if directory exists
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        async def root_fallback():
            return JSONResponse(
                content={
                    "message": "Welcome to Banking77 Intent Classification API",
                    "docs_url": "/docs",
                    "health_url": "/health",
                    "predict_url": "/predict",
                }
            )

    return app


app = create_app()
