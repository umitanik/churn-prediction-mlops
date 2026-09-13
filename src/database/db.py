import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import DateTime, Integer, Numeric, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. There is no default on purpose: a connection "
        "string carries credentials and does not belong in source code. "
        "See .env.example."
    )
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    """Declarative base.

    The schema is owned by Alembic - run ``alembic upgrade head`` to create or
    migrate it. There is deliberately no create_all() helper here: two owners
    for one schema means the outcome depends on which of them runs first.
    """


class PredictionLog(Base):
    __tablename__ = "customer_churn_prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    customer_id: Mapped[int] = mapped_column(Integer, nullable=True)
    surname: Mapped[str] = mapped_column(String(50), nullable=True)

    credit_score: Mapped[int] = mapped_column(Integer)
    geography: Mapped[str] = mapped_column(String(50))
    gender: Mapped[str] = mapped_column(String(10))
    age: Mapped[int] = mapped_column(Integer)
    tenure: Mapped[int] = mapped_column(Integer)
    balance: Mapped[float] = mapped_column(Numeric(12, 2, asdecimal=False))
    num_of_products: Mapped[int] = mapped_column(Integer)
    has_cr_card: Mapped[int] = mapped_column(Integer)
    is_active_member: Mapped[int] = mapped_column(Integer)
    estimated_salary: Mapped[float] = mapped_column(Numeric(12, 2, asdecimal=False))

    card_type: Mapped[str] = mapped_column(String(20), nullable=True)
    satisfaction_score: Mapped[int] = mapped_column(Integer, nullable=True)
    point_earned: Mapped[int] = mapped_column(Integer, nullable=True)

    model_version: Mapped[str] = mapped_column(String(64), nullable=True)

    prediction_label: Mapped[str] = mapped_column(String(20))
    churn_probability: Mapped[float] = mapped_column(Numeric(7, 6, asdecimal=False))

    actual_label: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    labeled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
