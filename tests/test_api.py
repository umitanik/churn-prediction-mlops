import pytest

PREDICTION_KEYS = {"prediction", "churn_probability", "log_id"}


def test_health_reports_a_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["database"] == "ok"


def test_predict_returns_the_documented_contract(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == PREDICTION_KEYS
    assert body["prediction"] in {"CHURN", "LOYAL"}
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_persists_the_prediction(client, payload):
    payload["Surname"] = "PersistenceCheck"
    log_id = client.post("/predict", json=payload).json()["log_id"]

    logs = client.get("/logs", params={"limit": 100}).json()
    stored = next(entry for entry in logs if entry["id"] == log_id)

    assert stored["surname"] == "PersistenceCheck"
    assert stored["geography"] == payload["Geography"]
    assert stored["prediction_label"] in {"CHURN", "LOYAL"}


@pytest.mark.parametrize(
    "field, values",
    [
        ("Geography", ["France", "Germany", "Spain"]),
        ("Gender", ["Female", "Male"]),
    ],
)
def test_categorical_input_changes_the_response(client, payload, field, values):
    """End-to-end version of the regression test, over HTTP.

    Before the fix these calls all returned the identical probability, because
    the categorical columns never survived preprocessing.
    """
    probabilities = set()
    for value in values:
        body = dict(payload)
        body[field] = value
        probabilities.add(client.post("/predict", json=body).json()["churn_probability"])

    assert len(probabilities) == len(values)


def test_unknown_category_is_a_client_error(client, payload):
    """422, not 500.

    The pipeline raises for an unseen category, but surfacing that as a server
    error would blame the service for the caller's bad input. The Literal types
    in the schema are what stop the request at the door.
    """
    payload["Geography"] = "Atlantis"

    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "field, value",
    [
        ("NumOfProducts", 99),
        ("SatisfactionScore", 9),
        ("SatisfactionScore", 0),
        ("Tenure", -1),
        ("Age", 5),
        ("CreditScore", 1000),
        ("Balance", -1.0),
        ("CardType", "BRONZE"),
        ("Gender", "Unspecified"),
    ],
)
def test_out_of_range_values_are_rejected(client, payload, field, value):
    """Bounds belong on the API, not only in the dashboard widgets.

    Anything can POST here directly, so the schema is the real contract.
    """
    payload[field] = value

    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field", ["SatisfactionScore", "PointEarned", "CardType"])
def test_missing_required_field_is_rejected(client, payload, field):
    """These three were absent from the schema and silently sent as zero."""
    del payload[field]

    assert client.post("/predict", json=payload).status_code == 422


def test_logs_do_not_leak_orm_internals(client, payload):
    client.post("/predict", json=payload)

    for entry in client.get("/logs").json():
        assert "_sa_instance_state" not in entry


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_logs_limit_is_bounded(client, limit):
    """An unbounded limit would let one request dump the whole table."""
    assert client.get("/logs", params={"limit": limit}).status_code == 422


def test_logs_respects_the_limit(client, payload):
    for _ in range(3):
        client.post("/predict", json=payload)

    assert len(client.get("/logs", params={"limit": 2}).json()) == 2


def test_delete_removes_the_log(client, payload):
    log_id = client.post("/predict", json=payload).json()["log_id"]

    assert client.delete(f"/logs/{log_id}").status_code == 200

    remaining = {entry["id"] for entry in client.get("/logs", params={"limit": 100}).json()}
    assert log_id not in remaining


def test_deleting_an_unknown_log_is_not_found(client):
    assert client.delete("/logs/999999").status_code == 404
