from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, require_api_key
from src.api.schemas import PredictionLogOut
from src.database.db import PredictionLog

router = APIRouter(tags=["History"], dependencies=[Depends(require_api_key)])


@router.get("/logs", response_model=list[PredictionLogOut])
def get_prediction_logs(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return (
        db.query(PredictionLog)
        .order_by(PredictionLog.id.desc())
        .limit(limit)
        .all()
    )
