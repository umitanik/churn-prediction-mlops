from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

API_TO_MODEL_COLUMNS = {
    "CardType": "Card Type",
    "SatisfactionScore": "Satisfaction Score",
    "PointEarned": "Point Earned",
}

class CustomerInput(BaseModel):
    CustomerId: int = Field(..., examples=[15634602])
    Surname: str = Field(..., examples=["Yilmaz"])
    CreditScore: int = Field(..., ge=300, le=850, examples=[619])
    Geography: Literal["France", "Germany", "Spain"] = Field(..., examples=["France"])
    Gender: Literal["Female", "Male"] = Field(..., examples=["Female"])
    Age: int = Field(..., ge=18, le=100, examples=[42])
    Tenure: int = Field(..., ge=0, le=10, examples=[2])
    Balance: float = Field(..., ge=0, examples=[0.0])
    NumOfProducts: int = Field(..., ge=1, le=4, examples=[1])
    HasCrCard: int = Field(..., ge=0, le=1, examples=[1])
    IsActiveMember: int = Field(..., ge=0, le=1, examples=[1])
    EstimatedSalary: float = Field(..., ge=0, examples=[101348.88])
    CardType: Literal["DIAMOND", "GOLD", "PLATINUM", "SILVER"] = Field(..., examples=["SILVER"])
    SatisfactionScore: int = Field(..., ge=1, le=5, examples=[3])
    PointEarned: int = Field(..., ge=0, examples=[500])

    def to_model_row(self) -> dict:
        """Return the payload keyed by the column names the model expects."""
        row = self.model_dump()
        for api_name, model_name in API_TO_MODEL_COLUMNS.items():
            row[model_name] = row.pop(api_name)
        return row


class PredictionResponse(BaseModel):
    prediction: Literal["CHURN", "LOYAL"]
    churn_probability: float
    log_id: int


class PredictionLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    customer_id: Optional[int]
    surname: Optional[str]
    credit_score: int
    geography: str
    gender: str
    age: int
    tenure: int
    balance: float
    num_of_products: int
    has_cr_card: int
    is_active_member: int
    estimated_salary: float
    prediction_label: str
    churn_probability: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_path: str
    database: str


class MessageResponse(BaseModel):
    message: str
