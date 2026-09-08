# Bank Customer Churn Prediction

A machine learning application designed to predict customer churn probability for banking institutions. This project uses a microservices architecture to serve a CatBoost model via a FastAPI backend, visualized through a Streamlit dashboard, with prediction logs stored in PostgreSQL.

## Architecture

The system is built as a set of connected microservices:
1.  **Frontend**: A [Streamlit](dashboard.py) interactive dashboard for users to input customer data and view risk analysis.
2.  **Backend (API)**: A [FastAPI](src/api/main.py) service that handles inference requests and persists every prediction.
3.  **Database**: A **PostgreSQL** instance for persisting `PredictionLog` entries (audit trail of all predictions).
4.  **ML Engine**: A reproducible pipeline using `CatBoost` and `scikit-learn` for the train/test split, feature engineering, encoding and training.

**Training happens outside the containers.** The serving image contains neither the
training data nor the model artifact — it only loads a model that was produced
beforehand. This keeps the serving container stateless and fast to start, and means every
replica serves the exact same model. The model reaches the container through the
`./models` bind mount.

## Getting Started

### 1. Set up the environment and train the model

```bash
uv venv --python 3.10 && uv pip install -r requirements.txt
```

```bash
uv run python -m src.ml.train
```

Training takes about half a minute. It creates `data/processed/` from the raw dataset if
it is missing, trains the CatBoost pipeline, prints the test metrics, and writes the
artifact to `models/catboost_churn_model.pkl`.

### 2. Create the database schema

```bash
docker compose up -d db
```

```bash
DATABASE_URL=postgresql://admin:654321@localhost:5433/churn_db uv run alembic upgrade head
```

Migrations are an explicit step, not something the API does on startup. The
application does not own the schema: a service that alters tables while several
replicas are starting produces a different result depending on which one wins.

### 3. Run the stack

```bash
docker compose up --build
```

Once running, access the services:
-   **Dashboard**: [http://localhost:8501](http://localhost:8501)
-   **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
-   **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

If you skip step 1 the backend **refuses to start** and the log tells you which command
to run; if you skip step 2 it starts but every request fails on a missing table. This is intentional: a service that cannot make predictions should fail loudly
rather than accept traffic and return `503` for every request.

## Database migrations

The schema is owned by Alembic. `src/database/db.py` deliberately has no
`create_all()` helper, because two owners for one schema means the result depends on
which of them runs first.

```bash
uv run alembic revision --autogenerate -m "describe the change"   # after editing a model
uv run alembic upgrade head                                       # apply
uv run alembic downgrade -1                                       # undo the last one
```

**Adopting Alembic on a database that already has data.** A database created before
these migrations existed has the table but no `alembic_version` row, so Alembic assumes
it is empty and tries to create the table again. Mark it as already migrated instead of
running the first revision:

```bash
uv run alembic stamp a7d6dc158ef1 && uv run alembic upgrade head
```

`stamp` records the revision without executing it; the following `upgrade` then applies
only what is genuinely missing, leaving existing rows untouched.

### Running without Docker

```bash
uv run uvicorn src.api.main:app --reload             # API on :8000
```

```bash
API_URL=http://127.0.0.1:8000 uv run streamlit run dashboard.py    # dashboard on :8501
```

The API needs a reachable PostgreSQL instance with the migrations applied. The compose
database is published on host port **5433** so it does not clash with a local PostgreSQL
on 5432:

```bash
DATABASE_URL=postgresql://admin:654321@localhost:5433/churn_db uv run uvicorn src.api.main:app --reload
```

Use `requirements-dev.txt` instead of `requirements.txt` if you also want to run the
notebooks or the experiment scripts (`src/ml/benchmark.py`, `src/ml/tuning.py`).

## Dataset

`data/raw/Customer-Churn-Records.csv` (10,000 rows, 18 columns) is tracked in this
repository and is the single source of truth.

`data/processed/` is **not** tracked. It is derived from the raw file by
[`src/data/split_dataset.py`](src/data/split_dataset.py) with a fixed seed
(`test_size=0.1`, `random_state=42`), which reproduces the same split byte for byte on
any machine:

```bash
uv run python -m src.data.split_dataset --force
```

## Machine Learning Pipeline

The ML workflow is designed for reproducibility and consistency between training and inference.

*   **Split**: [`src/data/split_dataset.py`](src/data/split_dataset.py) produces `train.csv`, `test.csv` (features only) and `ground_truth.csv` (labels only).
*   **Preprocessing**: [`src/ml/preprocessor.py`](src/ml/preprocessor.py) drops identifier columns and derives the engineered features. Every step here is row-wise, so a single row and a full batch are treated identically.
*   **Encoding**: categorical encoding is a fitted `OneHotEncoder` inside the pipeline, not a standalone `get_dummies` call. The categories are learned during training and travel with the artifact, so inference on a single row produces exactly the columns the model was trained on, and an unseen category raises instead of being silently scored as the reference category.
*   **Training**: `uv run python -m src.ml.train` trains the pipeline and saves it to `models/catboost_churn_model.pkl`. Add `--skip-if-exists` to make it a no-op when the artifact is already present.
*   **Model selection**: `uv run python -m src.ml.benchmark` compares eight classifiers with 5-fold cross-validation.
*   **Tuning**: `uv run python -m src.ml.tuning` runs Optuna searches for Gradient Boosting and CatBoost.
*   **Versioning**: training stamps the artifact with a UTC timestamp (`model_version_`), which the API reports on `/health` and writes to every prediction log, so a stored prediction can be attributed to the model that produced it.
*   **Experiments**: Check the `notebooks/` directory for exploratory data analysis and the tuning experiments.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Model version, model and database status. Returns `503` when degraded. |
| `POST` | `/predict` | Scores one customer and persists the result to `PredictionLog`. |
| `GET` | `/logs` | Returns the most recent prediction logs (`limit` between 1 and 100). |
| `DELETE` | `/logs/{log_id}` | Deletes a single prediction log. |

`/predict` requires all 15 customer fields, including `SatisfactionScore` and
`PointEarned`. Categorical fields (`Geography`, `Gender`, `CardType`) accept only the
values present in the training data; anything else is rejected with `422`.

## Project Structure

```
├── dashboard.py             # Streamlit frontend entry point
├── docker-compose.yaml      # Container orchestration
├── requirements.txt         # Runtime dependencies
├── requirements-dev.txt     # Notebook and experiment dependencies
├── alembic/                 # Database migrations (schema owner)
├── tests/                   # pytest suite
├── data/
│   ├── raw/                 # Source dataset (tracked)
│   └── processed/           # Derived train/test split (generated, not tracked)
├── models/                  # Trained model artifact (generated, not tracked)
├── notebooks/               # Jupyter notebooks for experimentation
└── src/
    ├── api/
    │   ├── main.py          # App, lifespan, router registration
    │   ├── schemas.py       # Request/response models
    │   ├── dependencies.py  # DB session, model and preprocessor providers
    │   └── routers/         # health, predictions, logs
    ├── data/                # Reproducible train/test split
    ├── database/            # Database connection & ORM models
    ├── ml/                  # Preprocessing, training, benchmarking, tuning
    └── utils/               # EDA and visualization helpers
```

## Key Technologies

*   **Language**: Python 3.10+
*   **Environment**: uv
*   **Web Frameworks**: FastAPI, Streamlit
*   **ML Libraries**: CatBoost, Scikit-learn, Pandas, Optuna
*   **Database**: PostgreSQL, SQLAlchemy
*   **Containerization**: Docker, Docker Compose
