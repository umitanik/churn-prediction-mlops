import pytest

from tests.conftest import TEST_API_KEY

PREDICTION_KEYS = {"prediction", "churn_probability", "decision_threshold", "log_id", "top_drivers"}


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


def test_nothing_in_the_api_accepts_delete(client):
    """An audit trail with a DELETE endpoint is not an audit trail.

    Checked structurally against the registered routes rather than by status
    code, so it holds no matter how the log endpoints are reorganised.
    """
    from src.api.main import app

    deletable = [r.path for r in app.routes if "DELETE" in getattr(r, "methods", set())]

    assert deletable == []


def test_health_is_open_without_a_key(client):
    """Health checks come from load balancers and Docker, which carry no key."""
    response = client.get("/health", headers={"X-API-Key": ""})

    assert response.status_code == 200


@pytest.mark.parametrize("path, method", [("/predict", "post"), ("/logs", "get")])
def test_missing_key_is_unauthorized(client, payload, path, method):
    request = getattr(client, method)
    kwargs = {"json": payload} if method == "post" else {}

    response = request(path, headers={"X-API-Key": ""}, **kwargs)

    assert response.status_code == 401


@pytest.mark.parametrize("path, method", [("/predict", "post"), ("/logs", "get")])
def test_wrong_key_is_unauthorized(client, payload, path, method):
    request = getattr(client, method)
    kwargs = {"json": payload} if method == "post" else {}

    response = request(path, headers={"X-API-Key": TEST_API_KEY + "x"}, **kwargs)

    assert response.status_code == 401


def test_unauthorized_request_writes_nothing(client, payload):
    """Rejected calls must not leave a trace in the audit log."""
    before = {e["id"] for e in client.get("/logs", params={"limit": 100}).json()}

    client.post("/predict", json=payload, headers={"X-API-Key": "wrong"})

    after = {e["id"] for e in client.get("/logs", params={"limit": 100}).json()}
    assert after == before


def test_prediction_explains_itself(client, payload):
    """Every prediction names the features that drove it."""
    drivers = client.post("/predict", json=payload).json()["top_drivers"]

    assert len(drivers) == 3
    for driver in drivers:
        assert set(driver) == {"feature", "contribution"}
        assert isinstance(driver["contribution"], float)


def test_drivers_are_ordered_by_influence(client, payload):
    drivers = client.post("/predict", json=payload).json()["top_drivers"]
    magnitudes = [abs(d["contribution"]) for d in drivers]

    assert magnitudes == sorted(magnitudes, reverse=True)


def test_drivers_name_real_model_features(client, payload):
    """A driver must be a column the model actually consumes."""
    from src.api.schemas import API_TO_MODEL_COLUMNS

    drivers = client.post("/predict", json=payload).json()["top_drivers"]
    raw_columns = set(payload) - {"CustomerId", "Surname"}
    known = ({API_TO_MODEL_COLUMNS.get(c, c) for c in raw_columns}
             | {"BalanceSalaryRatio", "TenureByAge", "HasBalance", "CreditScoreGivenAge"})

    for driver in drivers:
        base = driver["feature"].split("_")[0]
        assert driver["feature"] in known or base in known, driver["feature"]


def test_age_drives_an_older_inactive_customer(client, payload):
    """Sanity-check the explanation against what the EDA found.

    Age was the strongest single signal (r = 0.29) and inactive members churn
    more. An old, inactive customer should have Age among the top drivers, and
    it should push toward churn.
    """
    payload.update({"Age": 65, "IsActiveMember": 0})

    drivers = {d["feature"]: d["contribution"] for d in
               client.post("/predict", json=payload).json()["top_drivers"]}

    assert "Age" in drivers
    assert drivers["Age"] > 0


def test_health_reports_the_decision_threshold(client):
    body = client.get("/health").json()

    assert 0.0 < body["decision_threshold"] < 1.0


def test_label_follows_the_stored_threshold_not_a_hardcoded_half(client, payload):
    """The label is a business decision the artifact carries, not model.predict().

    Every prediction must agree with its own reported threshold. Scanning a
    range of ages finds customers on both sides of it; any probability in
    the open interval between the threshold and 0.5 exposes a label that
    silently reverted to the default.
    """
    seen_churn = seen_loyal = False
    for age in range(20, 80, 5):
        body = dict(payload, Age=age)
        response = client.post("/predict", json=body).json()
        expected = "CHURN" if response["churn_probability"] >= response["decision_threshold"] else "LOYAL"

        assert response["prediction"] == expected, response
        seen_churn |= response["prediction"] == "CHURN"
        seen_loyal |= response["prediction"] == "LOYAL"

    assert seen_churn and seen_loyal, "age sweep did not cross the threshold"
