import os
import sys
from contextlib import asynccontextmanager

import joblib
import uvicorn
from fastapi import FastAPI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.api.dependencies import MODEL_PATH
from src.api.routers import health, logs, predictions
from src.database.db import init_db
from src.ml.preprocessor import CustomerChurnPreprocessor


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("Database tables are ready.")

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model artifact not found: {MODEL_PATH}. "
            "Train it first with 'uv run python -m src.ml.train', "
            "then start the API again."
        )

    app.state.churn_model = joblib.load(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    app.state.preprocessor = CustomerChurnPreprocessor()
    print("Preprocessor initialized.")

    yield

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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
