import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

from src.database.db import SessionLocal

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(os.getcwd(), 'models', 'catboost_churn_model.pkl')
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_model(request: Request):
    model = getattr(request.app.state, "churn_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return model

def get_preprocessor(request: Request):
    preprocessor = getattr(request.app.state, "preprocessor", None)
    if preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor is not loaded.")
    return preprocessor

def get_model_version(request: Request) -> str:
    return getattr(request.app.state, "model_version", "unknown")

def get_decision_threshold(request: Request) -> float:
    return getattr(request.app.state, "decision_threshold", 0.5)

def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    expected = os.getenv("API_KEY", "")
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")
