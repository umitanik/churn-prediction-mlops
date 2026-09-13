# Bank Customer Churn Prediction

A machine learning application designed to predict customer churn probability for banking institutions. This project uses a microservices architecture to serve a CatBoost model via a FastAPI backend, visualized through a Streamlit dashboard, with prediction logs stored in PostgreSQL.

## Architecture

The system is built as a set of connected microservices:
1.  **Frontend**: A [Streamlit](dashboard.py) interactive dashboard for users to input customer data, view risk analysis, and record what actually happened.
2.  **Backend (API)**: A [FastAPI](src/api/main.py) service that scores customers, explains each score, and persists every prediction.
3.  **Database**: A **PostgreSQL** instance holding the prediction audit trail, including ground-truth outcomes once they are known.
4.  **ML Engine**: A reproducible pipeline using `CatBoost` and `scikit-learn` for the train/test split, feature engineering, encoding, training and threshold selection.

**Training happens outside the containers.** The serving image contains neither the
training data nor the model artifact — it only loads a model that was produced
beforehand. This keeps the serving container stateless and fast to start, and means every
replica serves the exact same model. The model reaches the container through the
`./models` bind mount.

## Getting Started

### 1. Configuration

```bash
cp .env.example .env
```

`.env` holds the database credentials, the host port the database is published on, and
the API key. Nothing sensitive is committed; `docker-compose.yaml` and the services read
everything from this file.

### 2. Environment and model

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. Each service
installs only what it runs (`ml`, `api`, `dashboard` extras); `--all-extras` is the
development setup.

```bash
uv sync --all-extras
```

```bash
uv run python -m src.ml.train
```

Training takes about two minutes: it creates `data/processed/` from the raw dataset if it
is missing, trains the CatBoost pipeline, chooses the decision threshold on out-of-fold
predictions, prints the test metrics at both the default and the chosen threshold, and
writes the artifact to `models/catboost_churn_model.pkl`.

### 3. Database schema

```bash
docker compose up -d db
```

```bash
set -a && source .env && set +a
DATABASE_URL=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_HOST_PORT/$POSTGRES_DB uv run alembic upgrade head
```

Migrations are an explicit step, not something the API does on startup. The
application does not own the schema: a service that alters tables while several
replicas are starting produces a different result depending on which one wins.

### 4. Run the stack

```bash
docker compose up --build
```

Once running, access the services:
-   **Dashboard**: [http://localhost:8501](http://localhost:8501)
-   **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs) — use **Authorize** with the key from `.env`
-   **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

If the model or the API key is missing the backend **refuses to start** and the log says
what is missing; if the migrations were skipped it starts but every request fails on a
missing table. This is intentional: a service that cannot make predictions should fail
loudly rather than accept traffic and return `503` for every request.

### Running without Docker

```bash
set -a && source .env && set +a
DATABASE_URL=postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@localhost:$POSTGRES_HOST_PORT/$POSTGRES_DB uv run uvicorn src.api.main:app --reload
```

```bash
set -a && source .env && set +a
API_URL=http://127.0.0.1:8000 uv run streamlit run dashboard.py
```

Both read `API_KEY` from the environment. Set `LOG_LEVEL=DEBUG` to see more from the API.

## Decision threshold

`model.predict()` labels a customer as churning when the probability exceeds 0.5. That
number is not a decision, it is the absence of one — and for churn the two mistakes are
not symmetric. A missed leaver costs the customer's future value; a needless retention
call costs one contact.

Training therefore chooses the threshold explicitly, on **out-of-fold** predictions from
the training set (never on the test set), by maximising F-beta with `beta = 2` — recall
weighted twice as heavily as precision. The chosen value is stored on the artifact
(`decision_threshold_`), the API applies it, reports it in every response and on
`/health`, and the dashboard displays it.

| Threshold | Precision (churn) | Recall (churn) | Churners caught (of 204) |
|---|---|---|---|
| 0.5 (default) | 0.77 | 0.49 | 100 |
| **0.130** (chosen) | 0.41 | **0.87** | **177** |

`--recall-weight` on `src.ml.train` changes the trade-off. `src/ml/threshold.py` also
contains `expected_net_benefit()`, a cost-based objective to swap in once the real contact
cost, retained value and campaign success rate are known — F-beta is the proxy until then.

## Explainability

Every `/predict` response carries the three features that moved this customer's score the
most, as SHAP values computed by CatBoost itself ([`src/ml/explain.py`](src/ml/explain.py)).
They are per-prediction, in log-odds, and satisfy `bias + sum(contributions) = logit(p)`
exactly. A positive value pushed the score toward churn.

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

## Dataset

`data/raw/Customer-Churn-Records.csv` (10,000 rows, 18 columns) is tracked in this
repository and is the single source of truth.

`data/processed/` is **not** tracked. It is derived from the raw file by
[`src/data/split_dataset.py`](src/data/split_dataset.py) with a fixed seed and a
stratified split (`test_size=0.1`, `random_state=42`, `stratify=Exited`), so the churn
rate is identical in both halves and the split reproduces byte for byte on any machine:

```bash
uv run python -m src.data.split_dataset --force
```

## Machine Learning Pipeline

*   **Split**: [`src/data/split_dataset.py`](src/data/split_dataset.py) produces `train.csv`, `test.csv` (features only) and `ground_truth.csv` (labels only).
*   **Preprocessing**: [`src/ml/preprocessor.py`](src/ml/preprocessor.py) drops identifier columns and derives the engineered features. Every step here is row-wise, so a single row and a full batch are treated identically.
*   **Encoding**: categorical encoding is a fitted `OneHotEncoder` inside the pipeline, not a standalone `get_dummies` call. The categories are learned during training and travel with the artifact, so inference on a single row produces exactly the columns the model was trained on, and an unseen category raises instead of being silently scored as the reference category.
*   **Training**: `uv run python -m src.ml.train` trains the pipeline, selects the threshold and saves both to `models/catboost_churn_model.pkl`. Add `--skip-if-exists` to make it a no-op when the artifact is already present.
*   **Versioning**: training stamps the artifact with a UTC timestamp (`model_version_`), which the API reports on `/health` and writes to every prediction log.
*   **Model selection**: `uv run python -m src.ml.benchmark` compares eight classifiers with 5-fold cross-validation.
*   **Tuning**: `uv run python -m src.ml.tuning` runs Optuna searches for Gradient Boosting and CatBoost.
*   **Experiments**: Check the `notebooks/` directory for exploratory data analysis and the tuning experiments.

## API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | — | Model version, decision threshold, model and database status. `503` when degraded. |
| `POST` | `/predict` | `X-API-Key` | Scores one customer, explains the score, persists the result. |
| `GET` | `/logs` | `X-API-Key` | The most recent prediction logs (`limit` between 1 and 100). |
| `POST` | `/feedback/{log_id}` | `X-API-Key` | Records the customer's real outcome on a stored prediction. |

Every endpoint except `/health` requires the `X-API-Key` header to match `API_KEY`;
health checks come from Docker and load balancers, which carry no key. A wrong or
missing key returns `401` and writes nothing.

There is no `DELETE`. The log is an audit trail of what the model told whom, and a
record that can be removed is not an audit record. Outcomes recorded through `/feedback`
are likewise written once: repeating the same value is a no-op, changing it returns `409`.

`/predict` requires the 13 customer feature fields (`CustomerId` and `Surname` are
optional and stored as `NULL` when absent). Categorical fields accept only the values
present in the training data; anything else is rejected with `422`.

```json
{
  "prediction": "CHURN",
  "churn_probability": 0.943172,
  "decision_threshold": 0.1304,
  "log_id": 42,
  "top_drivers": [
    {"feature": "Age",                 "contribution": 1.8889},
    {"feature": "IsActiveMember",      "contribution": 1.0663},
    {"feature": "CreditScoreGivenAge", "contribution": 0.5332}
  ]
}
```

## Closing the loop

Churn is only observable months after the prediction, so ground truth arrives through a
separate call. `POST /feedback/{log_id}` writes `actual_label` and `labeled_at` on the
stored row; the dashboard's history tab offers the same for unlabelled predictions. Every
row also keeps the full scored input and the `model_version` that produced it, so the
table can be turned back into a labelled dataset.

What this does **not** solve, and no schema can: if the bank acts on high-risk
predictions, the customers it retains will be labelled `LOYAL` *because* of the
intervention. Training on those labels teaches the model that high-risk customers stay.
A retraining programme needs a control group — a random slice of high-risk customers
deliberately left alone — to obtain unbiased outcomes.

## Tests

```bash
uv run pytest
```

74 tests in under a second, no mocks and no PostgreSQL: the suite uses the real model,
the real pipeline, the real HTTP layer and SQLite. The tests that matter most guard
train/serve parity — one row must score identically alone and inside a batch — and the
migrations, which are run for real and compared against the ORM models so a schema that
drifts is caught in CI rather than at deploy.

## Project Structure

```
├── dashboard.py             # Streamlit frontend entry point
├── docker-compose.yaml      # Container orchestration (reads .env)
├── pyproject.toml           # Project metadata and dependencies (extras: ml, api, dashboard)
├── uv.lock                  # Fully resolved dependency lock
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
    │   ├── dependencies.py  # DB session, model, threshold, API key
    │   └── routers/         # health, predictions, logs, feedback
    ├── data/                # Reproducible stratified split
    ├── database/            # SQLAlchemy models
    ├── ml/                  # preprocessing, training, threshold, explain, benchmark, tuning
    ├── utils/               # EDA and visualization helpers
    └── logging_config.py    # One logging setup for every entry point
```

## Key Technologies

*   **Language**: Python 3.10+
*   **Environment**: uv (`pyproject.toml` + `uv.lock`)
*   **Web Frameworks**: FastAPI, Streamlit
*   **ML Libraries**: CatBoost, Scikit-learn, Pandas, Optuna
*   **Database**: PostgreSQL, SQLAlchemy, Alembic
*   **Containerization**: Docker, Docker Compose
