import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.dependencies import (
    get_db,
    get_model,
    get_model_version,
    get_preprocessor,
)
from src.api.schemas import CustomerInput, PredictionResponse
from src.database.db import PredictionLog

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict_churn(
    customer: CustomerInput,
    db: Session = Depends(get_db),
    model=Depends(get_model),
    preprocessor=Depends(get_preprocessor),
    model_version: str = Depends(get_model_version),
):
    df = pd.DataFrame([customer.to_model_row()])

    try:
        features = preprocessor.preprocess(df)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
    except Exception as e:
        print(f"ERROR during inference: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

    result_label = "CHURN" if prediction == 1 else "LOYAL"

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
        churn_probability=float(probability),
    )

    db.add(new_log)
    db.commit()
    db.refresh(new_log)

    return PredictionResponse(
        prediction=result_label,
        churn_probability=round(float(probability), 4),
        log_id=new_log.id,
    )
