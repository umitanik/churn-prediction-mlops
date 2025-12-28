import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Float, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://admin:654321@localhost:5432/churn_db"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

def init_db():
    Base.metadata.create_all(bind=engine)

class PredictionLog(Base):
    __tablename__ = "customer_churn_prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=lambda: datetime.now(timezone.utc)
    )    
    customer_id: Mapped[int] = mapped_column(Integer, nullable=True)
    surname: Mapped[str] = mapped_column(String(50), nullable=True)

    credit_score: Mapped[int] = mapped_column(Integer)
    geography: Mapped[str] = mapped_column(String(50))
    gender: Mapped[str] = mapped_column(String(10))
    age: Mapped[int] = mapped_column(Integer)
    tenure: Mapped[int] = mapped_column(Integer)
    balance: Mapped[float] = mapped_column(Float)   
    num_of_products: Mapped[int] = mapped_column(Integer)
    has_cr_card: Mapped[int] = mapped_column(Integer)
    is_active_member: Mapped[int] = mapped_column(Integer)
    estimated_salary: Mapped[float] = mapped_column(Float)

    prediction_label: Mapped[str] = mapped_column(String(20)) 
    churn_probability: Mapped[float] = mapped_column(Float) 