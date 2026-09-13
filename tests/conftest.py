import os
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

os.environ.setdefault(
    "DATABASE_URL", f"sqlite:///{Path(tempfile.mkdtemp()) / 'test.db'}"
)
os.environ.setdefault("API_KEY", "test-key-not-a-secret")
TEST_API_KEY = os.environ["API_KEY"]

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from src.data.split_dataset import ensure_split
from src.ml import train as train_module
from src.ml.preprocessor import CustomerChurnPreprocessor

MODEL_PATH = PROJECT_ROOT / train_module.MODEL_DIR / train_module.MODEL_FILENAME

VALID_PAYLOAD = {
    "CustomerId": 15634602,
    "Surname": "Yilmaz",
    "CreditScore": 619,
    "Geography": "France",
    "Gender": "Female",
    "Age": 42,
    "Tenure": 2,
    "Balance": 85000.0,
    "NumOfProducts": 1,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 101348.88,
    "CardType": "SILVER",
    "SatisfactionScore": 3,
    "PointEarned": 500,
}


@pytest.fixture(scope="session", autouse=True)
def database_schema():
    """Create the test schema by running the real migrations.

    Using Alembic here rather than create_all() means the suite exercises the
    same path production uses, so a migration that no longer matches the models
    fails in CI instead of at deploy time.
    """
    from alembic import command
    from alembic.config import Config

    config = Config(str(PROJECT_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    command.upgrade(config, "head")


@pytest.fixture(scope="session")
def trained_model():
    """The serialized pipeline, trained on demand so a fresh clone can run pytest."""
    if not MODEL_PATH.exists():
        train_module.train()
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="session")
def preprocessor():
    return CustomerChurnPreprocessor()


@pytest.fixture(scope="session")
def train_frame():
    """The processed training set, without the target column."""
    train_path, _, _ = ensure_split()
    return pd.read_csv(train_path).drop(columns=["Exited"])


@pytest.fixture
def payload():
    """A fresh copy of a valid request body."""
    return dict(VALID_PAYLOAD)


@pytest.fixture(scope="session")
def client(trained_model):
    """A TestClient with the application lifespan actually run.

    Entering the context manager is what loads the model into ``app.state``;
    without it every request would fail the dependency check.
    """
    from src.api.main import app

    with TestClient(app, headers={"X-API-Key": TEST_API_KEY}) as test_client:
        yield test_client


def score(model, preprocessor, payload, **overrides):
    """Return the churn probability for one customer, as the API computes it."""
    from src.api.schemas import CustomerInput

    body = dict(payload)
    body.update(overrides)
    frame = pd.DataFrame([CustomerInput(**body).to_model_row()])
    return model.predict_proba(preprocessor.preprocess(frame))[0][1]


def unvalidated_frame(payload, **overrides):
    """Build a model input frame while skipping the Pydantic schema.

    The schema and the fitted encoder are two independent defences against a
    bad category. Tests that target the encoder have to get past the schema
    first, which is what this helper is for - it must not be used to model
    normal traffic.
    """
    from src.api.schemas import API_TO_MODEL_COLUMNS

    body = dict(payload)
    body.update(overrides)
    for api_name, model_name in API_TO_MODEL_COLUMNS.items():
        body[model_name] = body.pop(api_name)
    return pd.DataFrame([body])
