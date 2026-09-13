from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.api.dependencies import MODEL_PATH, get_db
from src.api.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response, db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception as e:
        database_status = f"error: {e}"

    model_loaded = (
        getattr(request.app.state, "churn_model", None) is not None
        and getattr(request.app.state, "preprocessor", None) is not None
    )
    healthy = model_loaded and database_status == "ok"

    if not healthy:
        response.status_code = 503

    return HealthResponse(
        status="ok" if healthy else "degraded",
        model_loaded=model_loaded,
        model_path=MODEL_PATH,
        model_version=getattr(request.app.state, "model_version", "unknown"),
        decision_threshold=getattr(request.app.state, "decision_threshold", 0.5),
        database=database_status,
    )
