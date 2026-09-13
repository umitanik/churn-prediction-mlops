import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import (
    get_db,
    get_decision_threshold,
    get_model,
    get_model_version,
    get_preprocessor,
    require_api_key,
)
from src.api.schemas import CustomerInput, PredictionResponse
from src.database.db import PredictionLog
from src.ml.explain import top_drivers

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"], dependencies=[Depends(require_api_key)])


@router.post("/predict", response_model=PredictionResponse)
def predict_churn(
    customer: CustomerInput,
    db: Session = Depends(get_db),
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor),
    model_version: str = Depends(get_model_version),
    threshold: float = Depends(get_decision_threshold),
):
    df = pd.DataFrame([customer.to_model_row()])

    try:
        features = preprocessor.preprocess(df)
        probability = float(model.predict_proba(features)[0][1])
        drivers = top_drivers(model, features)
    except Exception:
        logger.exception("Inference failed for customer_id=%s", customer.CustomerId)
        raise HTTPException(status_code=500, detail="Prediction failed.")

    result_label = "CHURN" if probability >= threshold else "LOYAL"

    new_log = PredictionLog(
        customer_id=customer.CustomerId,
        surname=customer.Surname,
        credit_score=customer.CreditScore,
        geography=customer.Geography,
        gender=customer.Gender,
        age=customer.Age,
        tenure=customer.Tenure,
        balance=customer.Balance,
        num_of_products=customer.NumOfProducts,
        has_cr_card=customer.HasCrCard,
        is_active_member=customer.IsActiveMember,
        estimated_salary=customer.EstimatedSalary,
        card_type=customer.CardType,
        satisfaction_score=customer.SatisfactionScore,
        point_earned=customer.PointEarned,
        model_version=model_version,
        prediction_label=result_label,
        churn_probability=probability,
    )

    db.add(new_log)
    db.commit()
    db.refresh(new_log)

    return PredictionResponse(
        prediction=result_label,
        churn_probability=round(probability, 6),
        decision_threshold=threshold,
        log_id=new_log.id,
        top_drivers=drivers,
    )
