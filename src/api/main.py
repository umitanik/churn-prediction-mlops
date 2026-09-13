import logging
import os
from contextlib import asynccontextmanager

import joblib
import uvicorn
from fastapi import FastAPI

from src.api.dependencies import MODEL_PATH
from src.api.routers import feedback, health, logs, predictions
from src.logging_config import configure_logging
from src.ml.preprocessor import CustomerChurnPreprocessor

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    if not os.getenv("API_KEY"):
        raise RuntimeError(
            "API_KEY is not set. Every endpoint except /health requires it; "
            "copy .env.example to .env and choose a value."
        )

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Train it first with 'uv run python -m src.ml.train', "
            "then start the API again."
        )

    app.state.churn_model = joblib.load(MODEL_PATH)
    app.state.model_version = getattr(app.state.churn_model, "model_version_", "unknown")
    app.state.decision_threshold = getattr(app.state.churn_model, "decision_threshold_", 0.5)
    if not hasattr(app.state.churn_model, "decision_threshold_"):
        logger.warning("Model artifact carries no decision threshold; using 0.5.")
    logger.info(f"Model loaded: {MODEL_PATH} (version {app.state.model_version}, threshold {app.state.decision_threshold:.4f})")

    app.state.preprocessor = CustomerChurnPreprocessor()
    logger.info("Preprocessor initialized.")

    yield

    app.state.model_version = None
    app.state.decision_threshold = None
    app.state.churn_model = None
    app.state.preprocessor = None


app = FastAPI(
    title="Bank Churn Prediction API",
    description="Single Model - Lifespan Structure",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(predictions.router)
app.include_router(logs.router)
app.include_router(feedback.router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
