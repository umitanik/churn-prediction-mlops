import os
import sys
import joblib
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.db import SessionLocal, init_db, PredictionLog
from src.ml.preprocessor import CustomerChurnPreprocessor



churn_model = None
preprocessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global churn_model, preprocessor
    
    try:
        init_db()
        print("Veritabanı tabloları hazır.")
    except Exception as e:
        print(f"Veritabanı hatası: {e}")

    model_path = os.path.join(os.getcwd(), 'models', 'catboost_churn_model.pkl')
    if os.path.exists(model_path):
        try:
            churn_model = joblib.load(model_path)
            print(f"Model yüklendi: {model_path}")
        except Exception as e:
            print(f"Model dosyası bozuk: {e}")
    else:
        print(f"Model dosyası bulunamadı: {model_path}")

    try:
        preprocessor = CustomerChurnPreprocessor()
        print("Preprocessor başlatıldı.")
    except Exception as e:
        print(f"Preprocessor hatası: {e}")

    yield
    churn_model = None
    preprocessor = None

app = FastAPI(
    title="Bank Churn Prediction API",
    description="Tek Model - Lifespan Yapısı",
    version="1.0.0",
    lifespan=lifespan
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
class CustomerInput(BaseModel):
    CustomerId: int = Field(..., json_schema_extra=15634602)
    Surname: str = Field(..., json_schema_extra="Yilmaz")
    CreditScore: int = Field(..., json_schema_extra=619, ge=300, le=850)
    Geography: str = Field(..., json_schema_extra="France")
    Gender: str = Field(..., json_schema_extra="Female")
    Age: int = Field(..., json_schema_extra=42, ge=18, le=100)
    Tenure: int = Field(..., json_schema_extra=2)
    Balance: float = Field(..., json_schema_extra=0.0)
    NumOfProducts: int = Field(..., json_schema_extra=1)
    HasCrCard: int = Field(..., json_schema_extra=1)
    CardType: str = Field(..., json_schema_extra={"example": "SILVER"}, description="GOLD, SILVER, DIAMOND, PLATINUM")
    IsActiveMember: int = Field(..., json_schema_extra=1)
    EstimatedSalary: float = Field(..., json_schema_extra=101348.88)
    
@app.post("/predict", tags=["Prediction"])
def predict_churn(customer: CustomerInput, db: Session = Depends(get_db)):
    if churn_model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model veya Preprocessor yüklü değil.")

    try:
        input_data = customer.model_dump()
        
        if 'CardType' in input_data:
            input_data['Card Type'] = input_data.pop('CardType')
            
        df = pd.DataFrame([input_data])

        df_processed = preprocessor.preprocess(df)
        
        try:
            expected_features = churn_model.feature_names_in_
        except AttributeError:
            try:
                expected_features = churn_model.feature_names_
            except AttributeError:
                expected_features = churn_model.steps[-1][1].feature_names_
        
        df_processed = df_processed.reindex(columns=expected_features, fill_value=0)
        
        prediction = churn_model.predict(df_processed)[0]
        probability = churn_model.predict_proba(df_processed)[0][1]

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
            prediction_label=result_label,
            churn_probability=float(probability)
        )
        
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        return {
            "prediction": result_label,
            "churn_probability": round(float(probability), 4),
            "log_id": new_log.id
        }

    except Exception as e:
        print(f"HATA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs", tags=["History"])
def get_prediction_logs(limit: int = 20, db: Session = Depends(get_db)):
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(limit).all()
        
        result = []
        for log in logs:
            log_dict = log.__dict__
            
            if "_sa_instance_state" in log_dict:
                del log_dict["_sa_instance_state"]
            
            result.append(log_dict)
            
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Loglar okunamadı: {str(e)}")

@app.delete("/logs/{log_id}", tags=["Management"])
def delete_log(log_id: int, db: Session = Depends(get_db)):
    log_to_delete = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    
    if log_to_delete is None:
        raise HTTPException(status_code=404, detail="Silinecek kayıt bulunamadı.")
    
    db.delete(log_to_delete)
    db.commit()
    
    return {"message": f"Log ID {log_id} başarıyla silindi."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)