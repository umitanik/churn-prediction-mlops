from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.dependencies import get_db
from src.api.schemas import MessageResponse, PredictionLogOut
from src.database.db import PredictionLog

router = APIRouter()


@router.get("/logs", response_model=list[PredictionLogOut], tags=["History"])
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


@router.delete("/logs/{log_id}", response_model=MessageResponse, tags=["Management"])
def delete_log(log_id: int, db: Session = Depends(get_db)):
    log_to_delete = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()

    if log_to_delete is None:
        raise HTTPException(status_code=404, detail="Record to delete was not found.")

    db.delete(log_to_delete)
    db.commit()

    return MessageResponse(message=f"Log ID {log_id} deleted successfully.")
