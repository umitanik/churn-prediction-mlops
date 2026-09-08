import os

from fastapi import HTTPException, Request

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
