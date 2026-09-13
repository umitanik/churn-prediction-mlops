from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import get_db, require_api_key
from src.api.schemas import FeedbackRequest, PredictionLogOut
from src.database.db import PredictionLog

router = APIRouter(tags=["Feedback"], dependencies=[Depends(require_api_key)])


@router.post("/feedback/{log_id}", response_model=PredictionLogOut)
def record_outcome(log_id: int, body: FeedbackRequest, db: Session = Depends(get_db)):
    log = db.get(PredictionLog, log_id)
    if log is None:
        raise HTTPException(status_code=404, detail="Prediction not found.")

    if log.actual_label is not None:
        if log.actual_label != body.actual_label:
            raise HTTPException(
                status_code=409,
                detail=f"Outcome already recorded as {log.actual_label}.",
            )
        return log

    log.actual_label = body.actual_label
    log.labeled_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(log)
    return log
