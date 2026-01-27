# API Deployment & Request Logging

## 1. Objective
The final stage is to serve the trained **CatBoost** model via a high-performance **FastAPI** application. This API not only provides real-time churn predictions but also includes a **database integration** to log every request and result for monitoring and auditing purposes.

## 2. Technology Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Web Framework** | **FastAPI** | Modern, fast web framework for building APIs with Python. |
| **Server** | **Uvicorn** | An ASGI web server implementation for Python. |
| **Database** | **SQLite** (via **SQLAlchemy**) | To persistently store prediction logs (History). |
| **Validation** | **Pydantic** | Data validation and settings management using Python type hints. |
| **Model** | **CatBoost (.pkl)** | Serialized machine learning pipeline. |

## 3. API Endpoints

### 1. Prediction Endpoint
*   **URL:** `/predict`
*   **Method:** `POST`
*   **Description:** Accepts customer data, executes the preprocessing & model pipeline, and saves the result to the database.
*   **Input (JSON):**
    ```json
    {
      "CustomerId": 15634602,
      "Surname": "Hargrave",
      "CreditScore": 619,
      "Geography": "France",
      "Gender": "Female",
      "Age": 42,
      "Tenure": 2,
      "Balance": 0.0,
      "NumOfProducts": 1,
      "HasCrCard": 1,
      "IsActiveMember": 1,
      "EstimatedSalary": 101348.88,
      "CardType": "DIAMOND"
    }
    ```
*   **Output (JSON):**
    ```json
    {
      "prediction": "CHURN",
      "churn_probability": 0.8521,
      "log_id": 45
    }
    ```

### 2. History Endpoint
*   **URL:** `/logs`
*   **Method:** `GET`
*   **Description:** Retrieves the history of past predictions stored in the database.
*   **Parameters:** `limit` (default: 20).

### 3. Management Endpoint
*   **URL:** `/logs/{log_id}`
*   **Method:** `DELETE`
*   **Description:** Deletes a specific log entry from the database.

## 4. Key Implementation Features

### Lifespan Event Handler
The application uses FastAPI's `lifespan` context manager to handle startup and shutdown events efficiently:
1.  **Startup:** Initializes the SQLite database tables and loads the Machine Learning model & Preprocessor into memory *once* (preventing reload latency).
2.  **Shutdown:** Cleans up resources.

### Database Integration
The system uses `SQLAlchemy` ORM to interact with the database. The `PredictionLog` table stores:
*   Input features (Age, Balance, etc.)
*   Model outputs (`prediction_label`, `churn_probability`)
*   Metadata (Client ID, Surname)

## 5. How to Run
To start the API server locally:
```bash
python src/api/API.py
```