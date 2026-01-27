# Bank Customer Churn Prediction

A machine learning application designed to predict customer churn probability for banking institutions. This project uses a microservices architecture to serve a CatBoost model via a FastAPI backend, visualized through a Streamlit dashboard, with prediction logs stored in PostgreSQL.

## Architecture

The system is built as a set of connected microservices:
1.  **Frontend**: A [Streamlit](dashboard.py) interactive dashboard for users to input customer data and view risk analysis.
2.  **Backend (API)**: A [FastAPI](src/api/API.py) service that handles inference requests, manages model lifecycle, and ensures consistent preprocessing.
3.  **Database**: A **PostgreSQL** instance for persisting `PredictionLog` entries (audit trail of all predictions).
4.  **ML Engine**: A robust pipeline using `CatBoost`, `scikit-learn`, and `pandas` for training and feature engineering.

## Getting Started

### Option 1: Docker (Recommended)

The easiest way to run the full stack (Database, API, and Frontend) is using Docker Compose.

```bash
# Build and start services
docker-compose up --build
```

Once running, access the services:
-   **Dashboard**: [http://localhost:8501](http://localhost:8501)
-   **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Local Development (Manual)

If you prefer running services individually for development:

**1. Database Setup**
Ensure you have PostgreSQL installed and running.
```bash
# Example connection string
export DATABASE_URL="postgresql://user:pass@localhost:5432/churn_db"
```

**2. Backend (FastAPI)**
Make sure the trained model exists in `models/` (run `src/ml/train.py` if needed).
```bash
# Install dependencies
pip install -r requirements.txt

# Start API
uvicorn src.api.API:app --reload
```

**3. Frontend (Streamlit)**
```bash
export API_URL="http://127.0.0.1:8000"
streamlit run dashboard.py
```

## Machine Learning Pipeline

The ML workflow is designed for reproducibility and consistency between training and inference.

*   **Training**: Run `python src/ml/train.py` to train the `CatBoost` model. The artifact is saved to `models/catboost_churn_model.pkl`.
*   **Preprocessing**: All feature engineering logic is centralized in [`src/ml/preprocessor.py`](src/ml/preprocessor.py). This ensures the API processes raw input exactly how the model expects it.
*   **Experiments**: Check the `notebooks/` directory for exploratory data analysis and model tuning experiments (numbered `01_...` to `06_...`).

## Project Structure

```
├── dashboard.py             # Streamlit frontend entry point
├── docker-compose.yaml      # Container orchestration
├── requirements.txt         # Project dependencies
├── models/                  # Trained model artifacts (.pkl)
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks for experimentation
└── src/
    ├── api/                 # FastAPI application
    ├── database/            # Database connection & ORM models
    └── ml/                  # ML training and preprocessing logic
```

## Key Technologies

*   **Language**: Python 3.9+
*   **Web Frameworks**: FastAPI, Streamlit
*   **ML Libraries**: CatBoost, Scikit-learn, Pandas
*   **Database**: PostgreSQL, SQLAlchemy
*   **Containerization**: Docker, Docker Compose